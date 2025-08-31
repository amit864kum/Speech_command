import matplotlib.pyplot as plt

def plot_fl_metrics(rounds, accuracy_list, loss_list):
    """
    Plots the accuracy and loss of the global model over federated learning rounds.
    
    Args:
        rounds (list): A list of round numbers (e.g., [1, 2, 3, ...]).
        accuracy_list (list): A list of accuracy values (0-100) for each round.
        loss_list (list): A list of loss values for each round.
    """
    # Create a figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Accuracy on the first subplot
    ax1.plot(rounds, accuracy_list, 'b-o', label='Accuracy')
    ax1.set_title('Global Model Accuracy per Round')
    ax1.set_xlabel('Federated Learning Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot Loss on the second subplot
    ax2.plot(rounds, loss_list, 'r-o', label='Loss')
    ax2.set_title('Global Model Loss per Round')
    ax2.set_xlabel('Federated Learning Round')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()