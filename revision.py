for epoch in range(100):
    
    # forward pass
    y_pred=model(x_train).sqeez()
    
    # Calcualte Loss
    loss=criteriron(y_pred,y_train)
    
    # Clear old gradients
    optimizer.zero_grad()
    
    # Calculate New Gradient
    loss.backward()
    
    # Update weight
    Optimizer.step