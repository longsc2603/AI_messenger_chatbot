import os, sys
from flask import Flask, request
from pymessenger import Bot
from chat import chat_bot

app = Flask("My bot")

FB_ACCESS_TOKEN = 'YOUR_TOKEN_GOES_HERE'
bot = Bot(FB_ACCESS_TOKEN)

VERIFICATION_TOKEN = "hello"

@app.route('/', methods = ['GET'])
def verify():
    # Webhook verification
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == "hello":
            return "Verification token mismatch", 403
        return request.args["hub.challenge"], 200
    return "Hello world", 200

@app.route('/', methods=['POST'])
def webhook():
	print(request.data)
	data = request.get_json()
	if data['object'] == "page":
		entries = data['entry']
		for entry in entries:
			messaging = entry['messaging']
			for messaging_event in messaging:
				sender_id = messaging_event['sender']['id']
				recipient_id = messaging_event['recipient']['id']
				if messaging_event.get('message'):
					# HANDLE NORMAL MESSAGES HERE
					if messaging_event['message'].get('text'):
						# HANDLE TEXT MESSAGES
						query = chat_bot(messaging_event['message']['text'])
						# ECHO THE RECEIVED MESSAGE
						bot.send_text_message(sender_id, query)
					if messaging_event['message'].get('attachments'):
						for att in messaging_event['message'].get('attachments'):
							if att['type'] == "image":
								attachment_url = att['payload']['url']
								bot.send_image_url(sender_id, attachment_url)	
								bot.send_text_message(sender_id, 'Did you just send me this image?')						
	return "ok", 200


if __name__ == "__main__":
    app.run(port=8000, use_reloader = True, debug=True)
