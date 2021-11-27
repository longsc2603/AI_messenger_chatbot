# A Facebook Messenger Chatbot using NLP
This project is about creating a messenger chatbot using basic NLP techniques and models like Logistic Regression, Naive Bayes or a simple neural network. The code is then deployed on Heroku server to run as a Facebook chatbot app.
# Installation:
* For creating Facebook app, please visit: https://developers.facebook.com/docs/messenger-platform/getting-started/app-setup/
* For creating a Heroku webhook, please visit: https://www.youtube.com/c/IndianPythonista/videos The videos in this channel are pretty clear.
* Setup virtual environment:

Use this command line: `python3 -m venv venv`   (the last term 'venv' is where you input the name you want)

Then, to activate it on Windows: `venv\Scripts\activate`
* Install the required packages:
After activating the virtual environment, use this command line to install all the packages specified:
`pip install -r requirements.txt`
* Pushing code to Heroku:
You can visit the git documentation page for clear instruction, or use these command in the given order:

`git add .`

`git commit -m your_commit_message`

`git push heroku master`
# Usage:
When you finish your installation, you can just simply chat with the bot, here is an example of what you should get when complete the steps above:

![image](https://user-images.githubusercontent.com/56826526/143668513-ca412e4c-6fbc-4f13-a774-7c53b48dd5cc.png)
![image](https://user-images.githubusercontent.com/56826526/143668490-d08ebe5c-6820-4c10-920d-ee67693efa3d.png)
# Customize for your own case:
I have already made some basic tags and responses for the bot, but if you ever feel the need to change the way the bot responses, you can always customize the [content.json](content.json) file just by creating some new `tag` and give possible `patterns` and `responses` for each `tag`.
