# TravelTime

Deep learning–based travel time estimation.

## Dataset Setup
1. **Unzip the dataset**  
   Extract the provided `VRPdata.zip` into a directory named `VRPdata`:

   ```bash
   unzip VRPdata.zip -d VRPdata
   ```

## Running the Code
2. **Move to the code directory**  
   Navigate into the `code` folder:

   ```bash
   cd code
   ```

3. **Run the experiment scripts**  
   Execute the following bash scripts **one at a time**:

   ```bash
   bash cmd1.test # This is to run the simulation
   bash cmd2.graph # This is to generate graphs after analyzing the results.
   bash cmd3.timecomplexity # This is to measure the computation time
   bash cmd4.rev1 # This is to add more graphs with RMSE and MAE performance metrics
   ```

Each script performs a specific experiment or analysis step. Review the script comments for details on required dependencies or expected outputs.
