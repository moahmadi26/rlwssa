#Dependencies
The scripts rely on stormpy in order to parse PRISM programs. The instructions to install stormpy and adding it to the pyton environment can be found here: https://moves-rwth.github.io/stormpy/installation.html


#Repository Structure
`src/` includes the scripts
`crns` contains the CRN models used for experimental evaluation of the framework. Each folder in this directory contains a PRISM file (.sm) describing the species and reactions in the CRN, and a JSON file including the path to the corresponding CRN file and the property that is being checked.

#Running the scripts
To run the scripts, after cloning the repository:

```
cd src/
python3 main.py ../crns/PATH_TO_JSON_FILE
```