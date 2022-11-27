from model import Model
from torch import nn

# change model_type parameter in order to load another pretrained model
model = Model()
model.initialize_data(train_foldername='train', test_foldername='test', train_batch_size=100, test_batch_size=439)
# change the learning rate and momentum using the method in order to update the optimizer
model.set_lr_momentum(0.015, 0.9)
# change the last layer the way you want
num_ftrs = model.model.AuxLogits.fc.in_features
model.model.AuxLogits.fc = nn.Linear(num_ftrs, model.output_num)
num_ftrs = model.model.fc.in_features
model.model.fc = nn.Linear(num_ftrs, model.output_num)

model.train(epochs=10)
# .pth extension is used for pytorch models
# this method saves the parameters only
model.save_model('Trained Model.pth')

# The code below loads a fine-tuned model to be used on a new dataset

# model_loaded = Model()
# call the initialize data method if training/testing data are changed

# model_loaded.initialize_data(train_foldername='train', test_foldername='test', train_batch_size=100, test_batch_size=439)
# model.set_lr_momentum(0.015, 0.9)

# When you load the parameters make sure you input the same last layer

# num_ftrs = model_loaded.AuxLogits.fc.in_features
# model_loaded.AuxLogits.fc = nn.Linear(num_ftrs, model_loaded.output_num)
# num_ftrs = model_loaded.fc.in_features
# model_loaded.fc = nn.Linear(num_ftrs, model_loaded.output_num)
# model_loaded.load_model('Trained Model.pth')
# model_loaded.test_model()




