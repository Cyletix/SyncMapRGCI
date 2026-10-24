#from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score
from typing import Any 
from pathlib import Path
import os


def to_categorical(y: Any, num_classes: int | None = None, dtype: str = "float32") -> np.ndarray[Any, Any]:
    """Converts a class vector (integers) to binary class matrix."""
    
    y_arr: np.ndarray[Any, Any] = np.array(y, dtype="int")
    input_shape = y_arr.shape
    y_flat = y_arr.ravel()
    
    if num_classes is None:
        if y_flat.size == 0:
            num_classes = 0 
        else:
            num_classes = int(np.max(y_flat)) + 1
            
    n = y_flat.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    
    if n > 0: 
        categorical[np.arange(n), y_flat] = 1
        
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class GraphWalkTest:
    
    # Fix: renamed G and A to lowercase g/a to avoid constant redefine errors
    # Fix: added [Any] generic for DiGraph
    time_delay: int
    time_counter: int
    g: nx.DiGraph
    true_label: np.ndarray[Any, Any]
    output_size: int
    a: np.ndarray[Any, Any]
    
    # Fix: simplify type to int since __init__ already casts
    output_class: int 
    previous_output_class: int | None
    previous_previous_output_class: int | None
    input_sequence: np.ndarray[Any, Any]
    input_class: np.ndarray[Any, Any]
    
    def __init__(self, time_delay: int, filename: str = "graph.dot"):
        '''
        '''
        self.time_delay = time_delay
        self.time_counter = 0

        # Load the graph structure.
        self._load_graph(filename, allow_resize=True)

        #random start    
        # Fix: cast to int to avoid np.integer vs int issues
        self.output_class = int(np.random.randint(self.output_size))# type: ignore
        self.previous_output_class = None
        self.previous_previous_output_class = None
        #self.plotGraph()
        #self.getInput()
        self.input_sequence = np.empty((0, self.output_size))
        self.input_class = np.empty(0)

    def _resolve_path(self, filename: str) -> str:
        """Resolve dataset path, mirroring __init__'s lookup logic."""
        from pathlib import Path
        fpath = Path(filename)
        if fpath.exists():
            return str(fpath)
        base = Path(__file__).resolve().parents[1]
        alt = base / "datasets" / filename
        if alt.exists():
            return str(alt)
        raise FileNotFoundError(f"Cannot locate dataset file: {filename}")

    def _load_graph(self, filename: str, allow_resize: bool) -> None:
        """
        Internal helper: load a .dot graph and update g / true_label / a.
        With allow_resize=False the output_size must stay the same; otherwise an exception is raised.
        """
        filename_resolved = self._resolve_path(filename)
        g_new = nx.DiGraph(nx.nx_agraph.read_dot(filename_resolved))

        label: Any = g_new.nodes(data="label")
        label_list: np.ndarray[Any, Any] = np.asarray(list(label))
        true_label_new = np.asarray([int(x) for x in label_list[:,1]]) - 1

        output_size_new = g_new.number_of_nodes()
        adj_matrix = nx.adjacency_matrix(g_new)
        a_new = np.array(adj_matrix.toarray(), dtype=np.float64)  # type: ignore

        for i in range(output_size_new):
            accum = a_new[i].sum()
            if accum != 0:
                a_new[i] = a_new[i] / accum
            else:
                print("ERROR: Node ", i, " without connections from found")
                exit()

        if hasattr(self, "output_size") and not allow_resize and output_size_new != self.output_size:
            raise ValueError(f"switch_dataset requires same output_size ({self.output_size}), got {output_size_new}")

        self.g = g_new
        self.true_label = true_label_new
        self.output_size = output_size_new
        self.a = a_new

    def switch_dataset(self, filename: str) -> None:
        """
        Switch to a new graph structure while keeping the output dimension.
        Used by the graph-switch experiment in the live view.
        """
        # Only graphs with the current output_size are allowed; otherwise the model weights cannot be reused.
        self._load_graph(filename, allow_resize=False)
        # Reset the walk state; output_size and the history statistics stay in the model.
        self.time_counter = 0
        self.output_class = int(np.random.randint(self.output_size))  # type: ignore
        self.previous_output_class = None
        self.previous_previous_output_class = None
        self.input_sequence = np.empty((0, self.output_size))
        self.input_class = np.empty(0)

    def trueLabel(self) -> np.ndarray[Any, Any]: 
        return self.true_label

    def get_true_adjacency_support(self) -> np.ndarray[Any, Any]:
        support = np.asarray(self.a > 0, dtype=int)
        np.fill_diagonal(support, 0)
        return support

    def is_effectively_directed(self) -> bool:
        support = np.asarray(self.get_true_adjacency_support(), dtype=int)
        return bool(not np.array_equal(support, support.T))
    
    def getOutputSize(self) -> int: 
        return self.output_size

    def updateTimeDelay(self) -> bool: 
        self.time_counter+= 1
        if self.time_counter > self.time_delay:
            self.time_counter = 0 
            self.previous_previous_output_class = self.previous_output_class
            self.previous_output_class = self.output_class
            return True
        else:
            return False

    def getInput(self, reset: bool = False) -> np.ndarray[Any, Any]: 
    
        update = self.updateTimeDelay()
    
        if update == True:
        
            #print(self.g)
            #print(a.shape)
            #transition= self.a[self.output_class]
            #print(len(transition))
            #print(transition.shape)
            #print(transition)
            self.previous_output_class = self.output_class
            
            self.output_class = int(np.random.choice(self.output_size ,1 ,p= self.a[self.output_class])[0])
        #print(choice)
        #print(a[0])
        #print(a[1])
        #print(a[2])
        
        noise_intensity = 0
        if self.previous_output_class is None or self.previous_output_class == self.output_class:
            input_value = to_categorical(self.output_class, self.output_size)*np.exp(-0.1*self.time_counter) + np.random.randn(self.output_size)*noise_intensity
        else:
            input_value = to_categorical(self.output_class, self.output_size)*np.exp(-0.1*self.time_counter) + np.random.randn(self.output_size)*noise_intensity + to_categorical(self.previous_output_class, self.output_size)*np.exp(-0.1*(self.time_counter+self.time_delay))
        
#        noise_intensity=0

#        if  self.previous_output_class is None or np.array_equal(self.previous_output_class, self.output_class):
#            input_value = self.output_class*np.exp(-0.1*self.time_counter) + np.random.randn(self.output_size)*noise_intensity
#        else:
#            if  self.previous_previous_output_class is None or np.array_equal(self.previous_previous_output_class, self.previous_output_class):
#                input_value = self.output_class*np.exp(-0.1*self.time_counter) + np.random.randn(self.output_size)*noise_intensity + self.previous_output_class*np.exp(-0.1*(self.time_counter+self.time_delay))
#            else:
#                input_value = self.output_class*np.exp(-0.1*self.time_counter) + np.random.randn(self.output_size)*noise_intensity + self.previous_output_class*np.exp(-0.1*(self.time_counter+self.time_delay)) + self.previous_previous_output_class*np.exp(-0.1*(self.time_counter+2.0*self.time_delay))
#    
        return input_value

        #for i in range():

        #print(a)
        #D = np.diag(np.sum(a, axis=0))
        #print(D)
        #T = np.dot(np.linalg.inv(D),a)
        #print(T)
        # let's evaluate the degree matrix D
        # ...and the transition matrix T
        #exit()
    
        
        #exit()
    
    def getSequence(self, sequence_size: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        #print(self.data.shape[0])
        #print(input_sequence.shape)
        #exit()
        self.input_sequence = np.empty((sequence_size, self.output_size))
        self.input_class = np.empty(sequence_size)
        
        for i in range(sequence_size):
            input_value = self.getInput()
            
            #input_class.append(self.chunk)
            #input_sequence.append(input_value)
            self.input_class[i] = self.output_class
            self.input_sequence[i] = input_value

        return self.input_sequence, self.input_class

    def plotGraph(self, save: bool = True) -> None: 

        options = {
            'node_size': 100,
            'arrowstyle': '-|>',
            'arrowsize': 12,
        }
        
        nx.draw_networkx(self.g, arrows=True, **options) # type: ignore

        if save:
            plt.savefig("results/picture/graph_plot.png")  # type: ignore
        
        plt.show() # type: ignore

    def plot(self, input_class: np.ndarray[Any, Any], input_sequence: np.ndarray[Any, Any] | None = None, save: bool = False) -> None: 
    
        a = np.asarray(input_class)
        t = [i for i, _ in enumerate(a)] 

        plt.plot(t, a) # type: ignore
        
        if input_sequence is not None: 
            sequence = [np.argmax(x) for x in input_sequence]
            plt.plot(t, sequence) # type: ignore

        if save == True:
            plt.savefig("results/picture/plot.png") # type: ignore
        
        plt.show() # type: ignore
        plt.close() # type: ignore
    
    def plotSuperposed(self, input_class: np.ndarray[Any, Any], input_sequence: np.ndarray[Any, Any] | None = None, save: bool = False) -> None: 
    
        if input_sequence is None:
            a_class = np.asarray(input_class)
            t = [i for i, _ in enumerate(a_class)]
            plt.plot(t, a_class) # type: ignore
        else:
            input_sequence_arr = np.asarray(input_sequence)
            t = [i for i, _ in enumerate(input_sequence_arr)] 

        #exit()

            for i in range(input_sequence_arr.shape[1]):
                a_plot = input_sequence_arr[:,i] # Rename variable to avoid clashing with outer 'a'
                plt.plot(t, a_plot) # type: ignore
            
            a_class = np.asarray(input_class)
            plt.plot(t, a_class) # type: ignore

        if save == True:
            plt.savefig("results/picture/plot.png") # type: ignore
        
        plt.show() # type: ignore
        plt.close() # type: ignore
    
    def evaluation(self, label: np.ndarray[Any, Any] | list[int]) -> float:
        return float(normalized_mutual_info_score(label, self.trueLabel()))
