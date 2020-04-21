# Aspect-based-sentiment-analysis
Deep learning based Sentiment analysis for different aspects

-----------------------Running train.py------------------------------------


optional arguments:

  -h,            --help            
		show this help message and exit
  
  -b  ,         --batch_size 

                      batch_size
                        
  -d ,         --dim    
  
                     embedding dimension
  
  -kf ,        --kfold    
  
                        number of fold
                        
  -i,          --epoch 

                        number of epochs
                        

required named arguments:

  -iskf,              --iskfold 

                             allow to train as kfold CV --give [True, False]
                        
  -train_path,      --train_path 

                              path to train data
                        
  -model_path ,      --model_path 

                             path to save the model
                        


-----------------------Running test.py---------------------------------------------

test.py -test_path [TEST_PATH] -model_path [MODEL_PATH]

required named arguments:

  -test_path,        --test_path

                           path to train data
  
  -model_path ,     --model_path  
 
                             path to save the model
  
  
