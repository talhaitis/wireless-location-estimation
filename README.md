# Wireless Location Estimation with Least Squares and Kalman Filter

## Description
This repository contains the implementation of various estimation techniques used for 2D position estimation based on range measurements to several fixed reference points. The lab explores the use of **Parametric Least Squares**, **Summation-of-Normals**, **Sequential Least Squares**, and **Kalman Filters** in estimating the position of a user in a wireless location system.

The lab uses **Time of Arrival (TOA)** range measurements to fixed targets for both stationary and moving user cases. The first 50 epochs represent stationary data, and after that, the user begins to move. Kalman filtering is applied to dynamically handle the moving user data, while the sequential least squares method is used for comparison.

The identification of stationary epochs was based on the observation that the position estimates for the first 50 epochs remained relatively constant. This allowed for a distinction between stationary data (where the user was not moving) and the subsequent moving data (where the position estimates changed over time). The **Kalman filter** was particularly useful for handling this dynamic data and improving position estimation as the user started moving after the first 50 epochs.

## Lab Objectives
- Review non-linear least-squares estimation techniques.
- Implement sequential least-squares and a basic Kalman filter.
- Understand the challenges of land-based wireless location using range measurements.
- Use Kalman filtering to handle dynamic data where sequential least squares may fail to provide accurate results.

## Installation & Requirements
1. Clone this repository:
    ```bash
    git clone https://github.com/talhaitis/wireless-location-estimation
    ```

2. Install dependencies:
    ```bash
    pip install numpy matplotlib
    ```

3. Place the `Lab1data.txt` file in the same directory as the script.

## How to Run
1. Run the main script:
    ```bash
    python lab1-refined.py
    ```

## Results

### 1. **Parametric Least Squares**
This method estimates the user's position based on range measurements from multiple targets using a non-linear least-squares approach. The user is initially stationary, and this method calculates the position for each epoch.

![X and Y Coordinates Over Time from Parametric Least Squares](results/positions_per_epoch.png)

### 2. **Batch Least Squares (BLS)**
In this approach, batch parametric least squares is applied to a set of measurements from stationary data to compute a more precise position estimate. The residuals for the range measurements are also analyzed to assess the accuracy of the batch solution.

![Residuals from Batch Least Squares](results/residual_from_BLS.png)

### 3. **Summation of Normal Least Squares (SONs)**
The Summation of Normal Least Squares method is applied as an alternative to batch least squares. It involves accumulating the normal equations over multiple epochs and then solving for the final position estimate.

![Residuals from Summation of Normal Least Squares](results/residual_from_SONs.png)

### 4. **Sequential Least Squares (SLs)**
This method processes the data sequentially, updating the position estimate after each epoch. However, as shown in the residuals plot, the method struggles to accurately track the user's movement after the 50th epoch, when the user begins to move.

![Residuals from Sequential Least Squares](results/residuals_from_SLs.png)

### 5. **Effect of Process Noise Q on Kalman Filter Solution**
This experiment shows how different values of process noise (Q) affect the Kalman filter's ability to track the user's position. Smaller values of Q provide smoother estimates, while larger values allow for more responsiveness to changes in the system.

![Effect of Process Noise Q on Kalman Filter Solution](results/process_noise_effct_KF.png)

### 6. **Kalman Filter with Constant Velocity Model**
The Kalman filter, with a constant velocity model, is applied to dynamically track the position of the user as they move. The filter adjusts the position estimate with each epoch, incorporating process noise and measurement uncertainty. The effect of varying the process noise (Q) is also demonstrated to show its impact on the Kalman filter solution.

![Estimated Position using Kalman Filter](results/estimated_position_EKF.png)


### 7. **Estimated Velocity Components Over Time KF**
In addition to tracking position, the Kalman filter also estimates the velocity components in both the X and Y directions. This plot demonstrates how the filter tracks the user's velocity over time, providing insights into their movement.

![Estimated Velocity Components](results/velocity_state_KF.png)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
