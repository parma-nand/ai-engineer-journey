for epoch in range(100):
    
  # Loss and Optimizer  
    criterion = nn.BCELoss()          # binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


#   Forward pass
y_pred=model(x_train).squeez()


# calculate loss
loss=criterion(y_pred,y_train)


# clear old gradient
optimizer.zero_grad()


# Claculate new gradient
loss.backward()


# update weighs
optimizer.step()
