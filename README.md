
# AI Pronunciation Trainer 
This tool uses AI (pretrained) intenting to be used as a tool to improve pronounciation.

## Installation
A conda env is suggested but not mandatory.

To use your local machine as a personal server,
```
pip install -r requirements.txt
python webApp.py
```
Install of ffmpeg is necessary.
```
sudo apt-get install ffmpeg
```
Also download from here https://ffmpeg.org/download.html. On Windows, it may be needed to add the ffmpeg "bin" folder to your PATH environment variable. On Mac, you can also just run "brew install ffmpeg".
A recent python 3.X version is needed.  

## Motivation (current author)

Just that I want to learn this beautiful language German. I am not claiming that I am not the major author but this is going to be enhansed significantly in the coming future.


## FAQ

### How do I add a new language?

There's definitely a code architecture that would allow this to be easier, but I have limited time and I think the current solution is doable with relative ease. What you need to do is: 
#### Backend 
As long as your language is supported by Whisper, you need only a database and small changes in key files:

1. Add your language identifier to the "lambdaGetSample.py" file
2. Add a .csv file with your text database in the "./databases" folder, following the naming convention 
3. Add a corresponding phonem model in the "RuleBasedModels.py", you likely need only to give the language code to Epitran and possibly correct some characters with a wrapper 
4. Add your trainer to "lambdaSpeechToScore.py" with the correct language code

If you language is not supported by Whisper, you need to have an Speech-To-Text model and add it to the "getASRModel" function in "models.py", and it needs to implement the "IASRModel" interface. Besides this, you need to do the same as above.
#### Frontend 

1. In the "callback.js" function, add a case for your language 
2. In the "main.html", add your language in the "languageBox" with corresponding call to the javascript function. 
