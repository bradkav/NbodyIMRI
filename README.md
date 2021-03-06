# NbodyIMRI

`NbodyIMRI` is a code for studying DM-dressed IMRI and EMRI systems using N-body simulations. The code calculates all BH-BH forces and BH-DM forces directly, while neglecting DM-DM pairwise interactions. This allows the code to scale up to very large numbers of DM particles, in order to study stochastic processes like dynamical friction.

#### Current version

**23/06/2022:** The current version is still in active development. However, there are some complete examples in the [`examples/`](examples) folder:

- [`Examples_Spike.ipynb`](examples/Examples_Spike.ipynb) - An example showing how to initialise a single BH surrounded by a Dark Matter spike and evolve in time.
- [`Examples_Binary.ipynb`](examples/Examples_Binary.ipynb) - A very simple example of an isolated binary, looking at how the time-step size affects the precision of the orbital evolution.
- [`Examples_Dressed_Binary.ipynb`](examples/Examples_Dressed_Binary.ipynb) - A more complicated example, for studying a BH binary embedded in a DM spike. This also includes an example of how to track events happening inside the code.

#### Installation

Install using 

```
pip install .
```

Alternatively, if you'd like to install the editable version, do:

```
pip install -e .
```


#### Structure

Note that all of the modules use `units.py` (included in the code) for specifying units and fundamental constants. The exception is `distributionfunctions.py`, which has it's own internal units system. 

The output directories for files can be changed by doing:

```python
import NbodyIMRI
NbodyIMRI.snapshot_dir = "path_to_directory"
```
