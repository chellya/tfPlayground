import tensorflow as tf
import numpy as np
from IPython.display import clear_output, Image, display, HTML
import webbrowser, os
import importlib

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph_def(graph_def,save_file=False, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))

    if save_file:
        with open('tf_graph.html', 'w') as f:
            f.write(HTML(iframe).data)
    else:
        display(HTML(iframe))

''' For Jupyter Notebook'''
def show_graph(graph=None):
    if graph is None:
        graph = tf.get_default_graph()
    show_graph_def(graph)

''' For none-Jupyter scenarios'''
def show_graph_local(graph=None):
    if graph is None:
        graph = tf.get_default_graph()
    show_graph_def(graph,save_file=True)
    webbrowser.open('file://' + os.path.realpath('tf_graph.html'))

print("sss")
v = 3