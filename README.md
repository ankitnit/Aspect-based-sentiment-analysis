# Aspect-based-sentiment-analysis
Deep learning based Sentiment analysis for different aspects

-----------------------Running train.py------------------------------------


optional arguments:

  -h, --help            show this help message and exit
  
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch_size
                        
  -d DIM, --dim DIM     embedding dimension
  
  -kf KFOLD, --kfold KFOLD
                        number of fold
                        
  -i EPOCH, --epoch EPOCH
                        number of epochs
                        

required named arguments:

  -iskf ISKFOLD, --iskfold ISKFOLD
                        allow to train as kfold CV --give [True, False]
                        
  -train_path TRAIN_PATH, --train_path TRAIN_PATH
                        path to train data
                        
  -model_path MODEL_PATH, --model_path MODEL_PATH
                        path to save the model
                        


-----------------------Running test.py---------------------------------------------

test.py -test_path [TEST_PATH] -model_path [MODEL_PATH]

required named arguments:

  -test_path, --test_path   [path to train data]
  
  -model_path , --model_path   [path to save the model]
  
  

