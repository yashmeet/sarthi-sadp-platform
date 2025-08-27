from google.cloud import pubsub_v1
import json
import structlog
from typing import Dict, Any, Optional
import asyncio
from concurrent.futures import TimeoutError

logger = structlog.get_logger()

class PubSubClient:
    """Google Cloud Pub/Sub client wrapper"""
    
    def __init__(self, settings):
        self.settings = settings
        self.publisher = None
        self.subscriber = None
        self.subscription_path = None
        self.topic_path = None
    
    async def connect(self):
        """Connect to Pub/Sub"""
        try:
            # Initialize publisher
            self.publisher = pubsub_v1.PublisherClient()
            self.topic_path = self.publisher.topic_path(
                self.settings.PROJECT_ID,
                self.settings.PUBSUB_TOPIC
            )
            
            # Initialize subscriber
            self.subscriber = pubsub_v1.SubscriberClient()
            self.subscription_path = self.subscriber.subscription_path(
                self.settings.PROJECT_ID,
                self.settings.PUBSUB_SUBSCRIPTION
            )
            
            logger.info("Connected to Pub/Sub",
                       topic=self.settings.PUBSUB_TOPIC,
                       subscription=self.settings.PUBSUB_SUBSCRIPTION)
            
        except Exception as e:
            logger.error(f"Failed to connect to Pub/Sub", error=str(e))
            raise
    
    async def publish_message(self, topic: str, message: Dict[str, Any]):
        """Publish message to topic"""
        try:
            # Prepare message
            message_data = json.dumps(message).encode('utf-8')
            
            # Get topic path
            topic_path = self.publisher.topic_path(
                self.settings.PROJECT_ID,
                topic
            )
            
            # Publish message
            future = self.publisher.publish(topic_path, message_data)
            message_id = future.result(timeout=10)
            
            logger.info(f"Published message",
                       topic=topic,
                       message_id=message_id)
            
            return message_id
            
        except TimeoutError:
            logger.error(f"Timeout publishing message", topic=topic)
            raise
        except Exception as e:
            logger.error(f"Failed to publish message", 
                        topic=topic,
                        error=str(e))
            raise
    
    async def subscribe(self, callback, max_messages: int = 10):
        """Subscribe to messages"""
        try:
            # Configure flow control
            flow_control = pubsub_v1.types.FlowControl(max_messages=max_messages)
            
            # Start pulling messages
            streaming_pull_future = self.subscriber.subscribe(
                self.subscription_path,
                callback=callback,
                flow_control=flow_control
            )
            
            logger.info(f"Started subscription",
                       subscription=self.settings.PUBSUB_SUBSCRIPTION)
            
            # Keep the main thread running
            with self.subscriber:
                try:
                    streaming_pull_future.result()
                except TimeoutError:
                    streaming_pull_future.cancel()
                    
        except Exception as e:
            logger.error(f"Subscription error", error=str(e))
            raise
    
    async def pull_messages(self, max_messages: int = 10) -> list:
        """Pull messages from subscription"""
        try:
            # Pull messages
            response = self.subscriber.pull(
                request={
                    "subscription": self.subscription_path,
                    "max_messages": max_messages,
                }
            )
            
            messages = []
            ack_ids = []
            
            for received_message in response.received_messages:
                # Decode message
                message_data = json.loads(
                    received_message.message.data.decode('utf-8')
                )
                messages.append(message_data)
                ack_ids.append(received_message.ack_id)
            
            # Acknowledge messages
            if ack_ids:
                self.subscriber.acknowledge(
                    request={
                        "subscription": self.subscription_path,
                        "ack_ids": ack_ids,
                    }
                )
                logger.info(f"Acknowledged {len(ack_ids)} messages")
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to pull messages", error=str(e))
            raise
    
    async def disconnect(self):
        """Disconnect from Pub/Sub"""
        try:
            if self.publisher:
                # Ensure all messages are published
                self.publisher.transport.close()
            
            if self.subscriber:
                self.subscriber.transport.close()
            
            logger.info("Disconnected from Pub/Sub")
            
        except Exception as e:
            logger.error(f"Error disconnecting from Pub/Sub", error=str(e))
            raise