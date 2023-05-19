# HELIOS-RL

This is the homepage of applying the HELIOS system to any Reinforcement Learning problem. 

The HELIOS approach is a generally applicable instruction following method whereby a two-layer hierarchy is formed: 1) a high level instruction plan and 2) the low level environment interaction. Uniquely, this work does not assume that instructions (or sub-goals) need to be supervised. Instead, we assume the environment contains some language such that this can be completed unsupervised. Furthermore, the unsupervised completion of each instruction is presented back to the user and their feedback strengthens the quality of the matching between observed environment positions and expected outcomes.

Instructions help greatly in mitigating the issue of long-term objectives never being reached and enable transfer of knowledge through the re-use of sub-instructions to new tasks. 

To make this work possible, we built this software solution such that it could enable the application of this work to any Reinforcement Learning problem. Unlike other Reinforcement Learning packages that are only designed to enable the importing a pre-built agent we go much further. 

First, we standardise the interaction loop setup such that setting up new problems is significantly easier and faster. 

Second, instead of simplying importing pre-built agents into a custom system we reverse this process so the interaction loop can be imported into far more complex hierarchical solutions without needing to be purpose built toeach problems. 

Lastly, analysis formatting and structure is generated such that the individual user only needs to interpret them to adjust parameters accordingly.

Provided a user can setup their problem using the template structure provided they can then leverage the most advanced Reinforcement Learning approaches with a simple parameter input. This also ensures the system is future-proof as new agents or encoders will be added as modules in later updates. 

A template for applying to any Reinforcement Learning problem can be found at the following link: https://github.com/pdfosborne/HELIOS-RL-TEMPLATE

Currently setup problems include:
- Classroom: A simple GridWorld educational problem https://github.com/pdfosborne/HELIOS-RL-Classroom 
- Chess: Play using the Python Chess engine re-defined as a Text Game https://github.com/pdfosborne/HELIOS-RL-Chess 
- ScienceWorld: Solve school level science problems in an interactive Text Game environment https://github.com/pdfosborne/HELIOS-RL-ScienceWorld
- TextWorldExpress: A collection of Text Games optimized to run fast! Includes CookingWorld, TextWorld Commonsense and Coin Collector https://github.com/pdfosborne/HELIOS-RL-TextWorldExpress