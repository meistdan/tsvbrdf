# Spatio-Temporal BRDFs
Source codes of the <a href="https://www.sciencedirect.com/science/article/abs/pii/S0097849321000431">Spatio-Temporal BRDFs</a> project. 
We used a modified version of <a href="https://github.com/jamriska/ebsynth">Ebsynth</a> for the synthesis.
We released also <a href="https://github.com/meistdan/mitsuba-tsvbrdf">a BSDF plugin</a> for the Mitsuba renderer in a separate repository.

## Usage
There is a sample material (SnowyGround) generated by the Substance Designer in data/original/SnowyGround.
There are scripts in the bin folder that you can use to run the synthesis:
```
bin/enlargement.sh # synthesizes larger material without guiding channel
bin/guided.sh # synthesizes material according a given target image (dafault is data/cag.png)
```
Regarding the guided synthesis, you can enable/disable 'correction' via the macro 'CORRECTION' in main.cpp. If it is enabled it adjust the resulting material to match the target material (see paper for the details).
In data/matlab, you can find MATLAB scripts that we used to preprocess and fit the data into polynomials.

## Dependencies
We use <a href="https://opencv.org/">OpenCV</a> for the data manipulation.
We compiled the project with Visual Studio 2015 (x64), but it should work also with other compilers using CMake.

## Data
We added one sample to the repository. We used data from <a href="https://www.cs.columbia.edu/CAVE/databases/staf/staf.php">the STAF database</a> and also synthetic data generated by Substance Designer. We may provide the whole dataset on demand.

## License
The code is released into the public domain. Note that the PatchMatch algorithm used in Ebsynth is patented by Adobe.

## Citation
If you use this code, please cite <a href="https://www.sciencedirect.com/science/article/abs/pii/S0097849321000431">the paper</a>:
```
@Article{Meister2021,
  author = {Daniel Meister and Adam Posp\'{\i}\v{s}il and Imari Sato and Ji\v{r}\'{\i} Bittner},
  title = {{Spatio-Temporal BRDF: Modeling and Synthesis}},
  journal = {Computers and Graphics},
  volume = {97},
  pages = {279-291},
  year = {2021},
}
```
