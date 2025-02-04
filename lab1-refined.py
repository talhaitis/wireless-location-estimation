import numpy as np
import matplotlib.pyplot as plt

# Read data function
def read_data(filePath):
    data = np.loadtxt(filePath, delimiter=',')
    return data
# Parametric Least Square
def parametric_least_square(data, target_coordinates):
    num_epochs = data.shape[0]
    num_targets = len(target_coordinates)

    # Initial estimate
    x0, y0 = 19, 30  

    positions = np.zeros((num_epochs, 2))

    for i in range(num_epochs):
        ranges = data[i, 1:]  
        A = np.zeros((num_targets, 2))
        w = np.zeros((num_targets, 1))

        for j in range(num_targets):
            dx = x0 - target_coordinates[j][0]
            dy = y0 - target_coordinates[j][1]
            r_computed = np.sqrt(dx**2 + dy**2)  
            
            A[j, 0] = dx / r_computed
            A[j, 1] = dy / r_computed
            w[j, 0] = r_computed - ranges[j]   

        delta = -np.linalg.pinv(A.T @ A) @ A.T @ w
        x0 += delta[0, 0]
        y0 += delta[1, 0]
        positions[i] = [x0, y0]

    return positions

# Batch Least Square
def batch_least_square(data, target_coordinates):
    stationary_ranges = data[:50, 1:]  
    num_epochs = stationary_ranges.shape[0]
    num_targets = len(target_coordinates)

    x0, y0 = 19, 30  

    A = np.zeros((num_targets * num_epochs, 2))
    w = np.zeros((num_targets * num_epochs, 1))

    row_index = 0  

    for i in range(num_epochs):
        ranges = data[i, 1:]  

        for j in range(num_targets):
            dx = x0 - target_coordinates[j][0]
            dy = y0 - target_coordinates[j][1]
            r_computed = np.sqrt(dx**2 + dy**2)  
            
            A[row_index, 0] = dx / r_computed
            A[row_index, 1] = dy / r_computed
            w[row_index, 0] = r_computed - ranges[j]  
            
            row_index += 1  

    delta = np.linalg.inv(A.T @ A) @ A.T @ w
    x0 += delta[0, 0]
    y0 += delta[1, 0]

    v_batch = w - A @ delta  
    posterior_variance_factor = (v_batch.T @ v_batch) / ((num_epochs * num_targets) - 2)

    return x0, y0, v_batch, posterior_variance_factor



def summation_of_normals(data, target_coordinates):

    # Use the first 50 epochs (4 ranges each)
    stationary_ranges = data[:50, 1:]
    num_epochs = stationary_ranges.shape[0]
    num_targets = len(target_coordinates)

    # Initial guess for position
    x0, y0 = 19, 30

    # Initialize accumulators for summation of normals
    N_total = np.zeros((2, 2))  # Normal matrix (sum of A^T A)
    u_total = np.zeros((2, 1))  # Right-hand side vector (sum of A^T w)
    v_list = []  

    # Loop over all epochs and accumulate normals
    for i in range(num_epochs):
        A = np.zeros((num_targets, 2))  # Design matrix
        w = np.zeros((num_targets, 1))  # Misclosure vector

        for j, (x_target, y_target) in enumerate(target_coordinates):
            dx = x0 - x_target
            dy = y0 - y_target
            r_computed = np.sqrt(dx**2 + dy**2)  # Computed range
            observed_range = stationary_ranges[i, j]  # Observed range

            A[j, 0] = dx / r_computed  # Partial derivative w.r.t x
            A[j, 1] = dy / r_computed  # Partial derivative w.r.t y
            w[j, 0] = observed_range - r_computed  # Misclosure vector

        # Compute normal matrix and right-hand side vector for this epoch
        N = A.T @ A
        u = A.T @ w

        # Accumulate normals
        N_total += N
        u_total += u

    # Solve for the final position estimate **AFTER accumulating all epochs**
    delta = np.linalg.inv(N_total) @ u_total
    x_final = x0 + delta[0, 0]
    y_final = y0 + delta[1, 0]

    # Compute residuals for each epoch using the final delta
    for i in range(num_epochs):
        A = np.zeros((num_targets, 2))
        w = np.zeros((num_targets, 1))

        for j, (x_target, y_target) in enumerate(target_coordinates):
            dx = x0 - x_target
            dy = y0 - y_target
            r_computed = np.sqrt(dx**2 + dy**2)  
            observed_range = stationary_ranges[i, j]  

            A[j, 0] = dx / r_computed  
            A[j, 1] = dy / r_computed  
            w[j, 0] = observed_range - r_computed  

        # Compute residuals using final delta
        residuals = w - A @ delta
        v_list.append(residuals)  # Store residuals per epoch

    # Stack residuals into a single array
    v_all = np.vstack(v_list)  # Residuals for all epochs

    # Compute posterior variance factor
    posterior_variance = (v_all.T @ v_all) / ((num_epochs * num_targets) - 2)

    return x_final, y_final, v_all, posterior_variance

def sequential_least_squares(ranges, target_coordinates):
    num_epochs, num_targets = ranges.shape

    # Initial guess for position
    x0, y0 = 19, 30
    x_hat = np.array([[x0], [y0]], dtype=np.float64)  

    # Initial covariance matrix (Assume identity scaled)
    C_x = np.eye(2) * 1000  # Large initial uncertainty

    C_l = np.eye(num_targets)  # Covariance matrix of measurements (identity for now)

    residuals = []  # Store residuals for all epochs

    for i in range(num_epochs):
        A = np.zeros((num_targets, 2))
        w = np.zeros((num_targets, 1))

        for j, (x_target, y_target) in enumerate(target_coordinates):
            dx, dy = x_hat[0, 0] - x_target, x_hat[1, 0] - y_target
            r_computed = np.sqrt(dx**2 + dy**2)
            observed_range = ranges[i, j]

            A[j] = [dx / r_computed, dy / r_computed]  # Jacobian
            w[j, 0] = observed_range - r_computed  # Misclosure (Innovation)

        # Compute Kalman Gain
        S = A @ C_x @ A.T + C_l  # Innovation covariance
        K = C_x @ A.T @ np.linalg.inv(S)  # Kalman gain

        # Update estimate
        x_hat += K @ w  # 

        # Update covariance
        C_x = (np.eye(2) - K @ A) @ C_x

        # Store residuals
        residuals.append(w)

    # Stack residuals into a single array
    v_all = np.vstack(residuals)

    # Compute posterior variance factor
    posterior_variance = (v_all.T @ v_all) / (num_epochs * num_targets - 2)

    return x_hat[0, 0], x_hat[1, 0], v_all, posterior_variance, C_x

def kalman_filter_with_random_walk(data, target_coordinates, q):
    num_epochs = data.shape[0]
    num_targets = len(target_coordinates)

    # Initial position estimate (x, y)
    x_hat = np.array([[19], [30]], dtype=np.float64)  

    # Initial covariance matrix (large uncertainty at the start)
    P = np.eye(2) * 1000  
    # Process noise matrix (Q = q * I)
    Q = np.eye(2) * q  

    # Measurement noise matrix (assuming identity for simplicity)
    R = np.eye(num_targets)  
    estimated_positions = np.zeros((num_epochs, 2))  
    for i in range(num_epochs):
        ranges = data[i, 1:]

        # Construct the measurement matrix H and Innovation vector S=(w)
        H = np.zeros((num_targets, 2))
        w = np.zeros((num_targets, 1))  

        for j, (x_target, y_target) in enumerate(target_coordinates):
            dx, dy = x_hat[0, 0] - x_target, x_hat[1, 0] - y_target
            r_computed = np.sqrt(dx**2 + dy**2)
            H[j] = [dx / r_computed, dy / r_computed]
            w[j, 0] = ranges[j] - r_computed  # Innovation (Residual)
         # Step 3: Prediction Step (Adding Process Noise)
        P = P + Q  
        # Step 4: Compute Kalman Gain
        S = H @ P @ H.T + R  # Innovation covariance
        K = P @ H.T @ np.linalg.inv(S)  # Kalman gain

        # Step 5: Update State Estimate
        x_hat = x_hat + K @ w  
        # Step 6: Update Covariance
        P = (np.eye(2) - K @ H) @ P  

        # Store estimated positions
        estimated_positions[i] = x_hat.flatten()

    return estimated_positions


def kalman_filter(data, target_coordinates, q, dt=1.0):
    """
    Implements a Kalman Filter with a Constant Velocity Model.
    
    Parameters:
        data (numpy array): The range measurements.
        target_coordinates (list of tuples): The positions of the reference points.
        q (float): Process noise parameter.
        dt (float): Time step between updates.

    Returns:
        estimated_states (numpy array): Estimated positions and velocities over time.
    """

    num_epochs = data.shape[0]  # Total number of epochs
    num_targets = len(target_coordinates)  # Number of reference points

    # Step 1: Initialize State Vector (Position & Velocity)
    x_hat = np.array([[19], [30], [0], [0]], dtype=np.float64)  # Initial (x, y, vx, vy)

    # Step 2: Initialize Covariance Matrix (P)
    P = np.eye(4) * 1000  # Large uncertainty for position and velocity

    # Step 3: Define Process Noise Matrix (Q)
    Q = np.eye(4) * q  

    # Step 4: Define Measurement Noise Matrix (R)
    R = np.eye(num_targets)  

    # Step 5: Define State Transition Matrix (F)
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Step 6: Store estimated positions and velocities
    estimated_states = np.zeros((num_epochs, 4))

    # Step 7: Kalman Filter Loop
    for i in range(num_epochs):
        ranges = data[i, 1:]

        # Step 7.1: Compute Measurement Matrix (H) & Residual (w)
        H = np.zeros((num_targets, 4))
        w = np.zeros((num_targets, 1))

        for j, (x_target, y_target) in enumerate(target_coordinates):
            dx, dy = x_hat[0, 0] - x_target, x_hat[1, 0] - y_target
            r_computed = np.sqrt(dx**2 + dy**2)
            H[j] = [dx / r_computed, dy / r_computed, 0, 0]
            w[j, 0] = ranges[j] - r_computed  # Innovation (Residual)

        # Step 7.2: Prediction Step
        x_hat = F @ x_hat  # Predict next state
        P = F @ P @ F.T + Q  # Predict covariance

        # Step 7.3: Compute Kalman Gain
        S = H @ P @ H.T + R  # Innovation covariance
        K = P @ H.T @ np.linalg.inv(S)  

        # Step 7.4: Update State Estimate
        x_hat = x_hat + K @ w  

        # Step 7.5: Update Covariance
        P = (np.eye(4) - K @ H) @ P  

        # Store estimated positions and velocities
        estimated_states[i] = x_hat.flatten()

    return estimated_states


    

def main():
    # Load data
    filePath = "Lab1data.txt"
    data = read_data(filePath)

    # Target coordinates
    target_coordinates = [(0, 0), (100, 0), (100, 100), (0, 100)]

    estimated_positions = parametric_least_square(data, target_coordinates)
    epochs = np.arange(len(estimated_positions))

    batch_x, batch_y, v_batch, posterior_variance_batch = batch_least_square(data, target_coordinates)

    print("Batch LS Estimated Position: X =", batch_x, ", Y =", batch_y)
    print("Batch LS Posterior Variance Factor:", posterior_variance_batch)

    # Call the Summation of Normal Least Squares function
    x_final, y_final, v_all, posterior_variance = summation_of_normals(data, target_coordinates)

    # Print results
    print(f"Summation of Normal LS Estimated Position: X = {x_final:.3f}, Y = {y_final:.3f}")
    print(f"Summation of Normal LS Posterior Variance Factor: {posterior_variance[0, 0]:.6f}")

  


    # Plot estimated positions from parametric least square
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, estimated_positions[:, 0], marker='o', linestyle='-', label="X Coordinate", color="b")
    plt.plot(epochs, estimated_positions[:, 1], marker='s', linestyle='-', label="Y Coordinate", color="r")
    plt.xlabel("Epoch")
    plt.ylabel("Coordinate Value")
    plt.title("X and Y Coordinates Over Time from Parametric Least Squares")
    plt.grid()
    plt.legend()
    plt.show()

    # 
     # Plot residuals from Batch Least Squares
    plt.figure(figsize=(10, 5))
    plt.plot(v_batch[::4], marker='o', linestyle='-', color="b", label="Batch LS Residuals")
    plt.xlabel("Epoch")
    plt.ylabel("Residual Value")
    plt.title("Residuals from Batch Least Squares")
    plt.grid()
    plt.legend()
    plt.show()

    # Plot residuals from Summation of Normal Least Squares
    plt.figure(figsize=(10, 5))
    plt.plot(v_all[::4], marker='s', linestyle='-', color="r", label="Summation LS Residuals")
    plt.xlabel("Epoch")
    plt.ylabel("Residual Value")
    plt.title("Residuals from Summation of Normal Least Squares")
    plt.grid()
    plt.legend()
    plt.show()

    # Call Sequential Least Squares
    seq_x, seq_y, v_seq, posterior_variance_seq, C_x_seq = sequential_least_squares(data[:50, 1:], target_coordinates)

    print(f"Sequential LS Estimated Position: X = {seq_x:.3f}, Y = {seq_y:.3f}")
    print(f"Sequential LS Posterior Variance Factor: {posterior_variance_seq[0, 0]:.6f}")

    # Plot residuals from Sequential LS
    plt.figure(figsize=(10, 5))
    plt.plot(v_seq[::4], marker='s', linestyle='-', color="g", label="Sequential LS Residuals")
    plt.xlabel("Epoch")
    plt.ylabel("Residual Value")
    plt.title("Residuals from Sequential Least Squares")
    plt.grid()
    plt.legend()
    plt.show()

    seq_x, seq_y, v_seq, posterior_variance_seq, C_x_seq = sequential_least_squares(data[:, 1:], target_coordinates)
    # Call Sequential Least Squares

    print(f"Sequential LS Estimated Position: X = {seq_x:.3f}, Y = {seq_y:.3f}")
    print(f"Sequential LS Posterior Variance Factor: {posterior_variance_seq[0, 0]:.6f}")

    # Plot residuals from Sequential LS
    plt.figure(figsize=(10, 5))
    plt.plot(v_seq[::4], marker='s', linestyle='-', color="g", label="Sequential LS Residuals Full")
    plt.xlabel("Epoch")
    plt.ylabel("Residual Value")
    plt.title("Residuals from Sequential Least Squares")
    plt.grid()
    plt.legend()
    plt.show()

        # Step 1: Experiment with different process noise values
    q_values = [0, 0.1, 10]  # Small, moderate, large process noise

    # Step 2: Plot results
    plt.figure(figsize=(10, 5))

    for q in q_values:
        estimated_positions = kalman_filter_with_random_walk(data, target_coordinates, q)
        plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], marker='o', linestyle='-', label=f"q = {q}")

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Effect of Process Noise Q on Kalman Filter Solution")
    plt.legend()
    plt.grid()
    plt.show()


 # Run Kalman Filter with constant Q
    q = 1.0  # Adjust process noise value
    estimated_states = kalman_filter(data, target_coordinates, q)

    # Extract position and velocity estimates
    epochs = np.arange(estimated_states.shape[0])
    x_positions = estimated_states[:, 0]
    y_positions = estimated_states[:, 1]
    x_velocities = estimated_states[:, 2]
    y_velocities = estimated_states[:, 3]

    # Plot Estimated Position
    plt.figure(figsize=(10, 5))
    plt.plot(x_positions, y_positions, marker='o', linestyle='-', label="Estimated Position")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Estimated Position using Kalman Filter with Velocity Model")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Estimated Velocity Components
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, x_velocities, marker='o', linestyle='-', label="Velocity X", color="b")
    plt.plot(epochs, y_velocities, marker='s', linestyle='-', label="Velocity Y", color="r")
    plt.xlabel("Epoch")
    plt.ylabel("Velocity (m/s)")
    plt.title("Estimated Velocity Components Over Time")
    plt.legend()
    plt.grid()
    plt.show()




if __name__ == "__main__":
    main()

