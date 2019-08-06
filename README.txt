*************************************************************************************************************************
     *	Project Title: Classification and Quantification of Plasmodium Parasites Using Convolutional Neural Networks	
     *	Author:        Maxwell Aladago						   					
     *	Supervisor:    Dr. Elena Rosca / Dr. Lorenzo Torresani						   		
    	                                                       
*************************************************************************************************************************
					Project Summary
	
This capstone project was an undergraduate thesis which researched about the detection of the detection of the 		
growth-cycle stages(gametocytes and trophozoites) of Plasmodium in Giemsa-stained thin blood smears. The study used 	
convolutional neural networks along with other novel deep learning techniques such data augmentation. Finally,the 	
detection happened by semantic segementation. 										

*************************************************************************************************************************

					Installations

1. Install Python 3.5 or higher												
2. Install other packages listed in the file requirements.txt at the root folder.					
if using pip, type the line below on a terminal and press enter								

		pip install -r requirements.txt 


Note: if running on a GPU (explained below), replace requirements.txt above with requirements-gpu.txt			

*************************************************************************************************************************	
					System Requirements

Note: The scripts can only be run given a minium of 30GB memory. Running the scripts on a lower memory machine will lead
to out of memory error and can damage the machine. 

1. The experiements were conducted using a Google Cloud Virtual Instance with a GPU and a 30GB memory. If you cannot 		
afford enough memory locally, you can set up a Google Virtual Machine. GPU is not required to run the scripts but it 	
will speed up both training and evaluation enormously. Once again, DO NOT run the scripts on a machine with less than	 
30GB memory.														

*************************************************************************************************************************
				  	CODE
1. All scripts for the growth-cycle classification can be found at ./GCC/code/						
2. Scripts for another experiment(binary classification) are at ./BC/							

*************************************************************************************************************************
					Running the Training Modules

1. To run the training module for the growth-cycle stage classification, launch a command prompt navigate to		
	./GCC/code . Once at the code directory in the command prompt, type 						


		python semantic_segmentor.py		


and press the enter key. 												

Note: The script has many command line arguments which can be used to control the execution of the script. Inspect the	 

./GCC/code/semantic_segmentor.py to see which all the arguments. To specify a command line arguement type		


		python semantic_segmentor.py --argument type value (eg. python semantic_segmentor.py --lr 0.001)	


Training will proceed without any command line argument. 								




2. To run the training module for the binary classification module, lauch a command prompt and navigate to 		
	./BC/. Once on that BC directory in the command prompt, type							

	
		python train.py


Like the growth cycle stage module,the train.py file has command line arguments which can be used run the models	
under different states. 												

NOTE: Change python in the commands above to python3 if you are not using a virtual environment. 			

*************************************************************************************************************************
					Running the Testing Modules

1. To run the test for the growth-cycle classification modle, open a command prompt and navigate to ./GCC/code . Once 	
at the code directory in the terminal, type										


		python evaluate.py

and press the return key


2. To run the test for the binary classification model, open a command prompt and navigate to ./BC/ . Once at that 	
directory type 
															
		python evaluate.py

and press the return key												


Like the training models, the evaluate.py files have command line arguments which can be used to change for instance 	
the model used to evaluate. 												

*************************************************************************************************************************

				Other Things to Note

1. The folder contains other folders. These folders mostly contain data and are referenced by scripts. If there is the	 
need to rename a folder or change its location, make the necessary corrections in the scripts accordingly.		 

2. The ./GCC/code/ and ./BC/ directories have other python scripts besides those specified above. Most of the these 	
scripts are data pre-processing scripts and do not directly influence the execution of the training nor testing modules.
Inspection of these scripts, however, will result in a much better understanding of the structure of the program	

3. Other folders contain only the results of some of the experiments. 

***************************************************************************************************************************




