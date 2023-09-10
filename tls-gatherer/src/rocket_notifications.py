import requests
import json


class RocketNotifications():
    def __init__(self, logging, webhook):
        """initializes webhook object"""
        self.url = 'https://ano/hooks/' + webhook
        self.logger = logging

    def send(self, text):
        """sends rocketchat notification"""

        data = {"alias": "Data Collection Update","text": text}
        # r = requests.post(self.url, json.dumps(data)).content
        #if not json.loads(r)['success'] == True:
        #    self.logger.warn("Failed to send Rocketchat Notification.")
