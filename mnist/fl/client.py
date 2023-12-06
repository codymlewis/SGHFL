class Client:
    def __init__(self, data, create_model_fn, seed=None):
        self.data = data
        self.model = create_model_fn(seed=seed)

    def fit(self, parameters, config):
        """
        Perform an local training.

        Arguments:
        - parameters: Global model parameters to start training from
        - config: Federated learning configuration object holding auxillary training specifications
        """
        self.model.set_parameters(parameters)
        metrics = self.model.step(
            self.data['train']['X'],
            self.data['train']['Y'],
            epochs=config['num_epochs'],
            batch_size=bs if (bs := config.get('batch_size')) else 32,
            steps_per_epoch=config.get("num_steps"),
            verbose=config.get("verbose")
        )
        return self.model.get_parameters(), len(self.data['train']), metrics

    def evaluate(self, parameters, config):
        """
        Evaluate the global model performance on this client's test dataset.

        Arguments:
        - parameters: Global model parameters to start training from
        - config: Federated learning configuration object holding auxillary training specifications
        """
        self.model.set_parameters(parameters)
        metrics = self.model.evaluate(
            self.data['test']['X'], self.data['test']['Y'],
            batch_size=bs if (bs := config.get('batch_size')) else 32,
            verbose=config.get("verbose")
        )
        return len(self.data['test']), metrics
