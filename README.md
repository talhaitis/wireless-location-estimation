# Wireless Location Estimation with Least Squares and Kalman Filter

## Description
This repository contains the implementation of various estimation techniques to estimate the position of a user based on range measurements to several fixed reference points. The lab focuses on 2D position estimation using **Parametric Least Squares**, **Summation-of-Normals**, **Sequential Least Squares**, and a simple **Kalman Filter**. The lab explores these methods with TOA/TDOA (Time of Arrival / Time Difference of Arrival) data for land-based wireless location systems.

## Lab Objectives
- Review non-linear least-squares estimation techniques.
- Implement sequential least-squares and a basic Kalman filter.
- Gain insights into the challenges of land-based wireless location using range measurements.

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
The script will perform the following:
- Estimate the 2D position for each epoch using parametric least squares.
- Plot the results of the position estimates over time.
- Compute and plot the residuals for different least squares methods.
- Implement and visualize the effect of process noise on a Kalman filter solution.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
