import torch


def generate_linear_dataset(
        n_samples_tot,
        n_samples_val,
        x_min,
        x_max,
        m_real,
        q_real,
        sigma_real
    ):
    """
    Generates a datset of points in the [x_min, x_max] range normally
    distributed around the straight line
        y = m_real * x + q_real
    with standard deviation `sigma_real`. The dataset consists of
    `n_samples_tot` points randomly split into
    `(n_samples_tot - n_samples_val)` training points and `n_samples_val`
    validation points.
    """
    # Build total (training + validation) dataset.
    x = torch.unsqueeze(
        torch.rand(n_samples_tot) * (x_max - x_min) + x_min,
        axis=-1
    )  # Shape: (n_samples_tot, 1).
    y = (
        m_real * x[:, 0]
        + q_real
        + torch.normal(0., sigma_real, (n_samples_tot,))
    )  # Shape: (n_samples_tot,).

    # Randomly shuffle the indices.
    indices = torch.randperm(x.shape[0])
    x = x[indices, ...]
    y = y[indices]

    x_val, y_val = x[:n_samples_val, ...], y[:n_samples_val]
    x_train, y_train = x[n_samples_val:], y[n_samples_val:]

    return x_train, y_train, x_val, y_val


def training_step_optimizer(
        training_data,
        val_data,
        model,
        loss_fn,
        optimizer,
        **kwargs
    ):
    """
    One training step for the parameters in `model` using `training_data`.
    """
    x_train, y_train = training_data
    x_val, y_val = val_data

    # Compute training loss.
    if 'params' in kwargs:
        y_pred = model(x_train, *kwargs['params'])
    else:
        y_pred = model(x_train)

    loss = loss_fn(y_pred, y_train)

    # Compute validation loss.
    with torch.no_grad():
        if 'params' in kwargs:
            y_pred_val = model(x_val, *kwargs['params'])
        else:
            y_pred_val = model(x_val)

        val_loss = loss_fn(y_pred_val, y_val)

    # Reset the gradients.
    optimizer.zero_grad()

    # Compute gradient.
    loss.backward()

    # Perform an optimization step.
    optimizer.step()

    return float(loss.detach().numpy()), float(val_loss.numpy())
