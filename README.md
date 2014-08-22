
## Quantification of damage to vasculature from microelectrode insertion ##

### overview ###

This package supplies software tools for the quantification of vascular damage
and of signal quality from extracellular electrophysiological recordings. 

Soon, it will also provide completely a completely automated script for 
construction of the local vasculature model through Vaa3d. 

#### dependencies ####
* python (written for 2.x)
* numpy (written with 1.8.1)
* scipy (written with 0.14.0)
* matplotlib (written with 1.3.1)

##### future #####
* vaa3d

### operation ###

1. run damage-quant, see damage-quant -h for syntax
2. place probe representation where you judge the insertion occurred in the 
   post stack
3. register the pre (green) and post (red) stacks to each other by xy-plane 
   rotation, xy-plane translation, and z-offsetting
