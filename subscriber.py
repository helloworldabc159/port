import paho.mqtt.client as mqtt
import time
# 定义全局变量
received_message = None

class MqttRoad(object):

    def __init__(self, mqtt_host, mqtt_port, mqtt_keepalive):
        super(MqttRoad, self).__init__()
        self.client = mqtt.Client()
        self.message = None
        self.received_message = False  # 标志位

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_publish = self.on_publish
        self.client.on_subscribe = self.on_subscribe
        self.client.on_unsubscribe = self.on_unsubscribe
        self.client.on_disconnect = self.on_disconnect

        self.client.connect(mqtt_host, mqtt_port, mqtt_keepalive)  # 600为keepalive的时间间隔
        self.client.loop_start()  # 在后台运行

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code: " + str(rc))
        # 订阅
        client.subscribe("mqtt11")

    def on_message(self, client, userdata, msg):
        global received_message
        received_message = str(msg.payload.decode('utf-8'))
        print("on_message topic:" + msg.topic + " message:" + str(msg.payload.decode('utf-8')))

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("On Subscribed: qos = %d" % granted_qos)

    def on_unsubscribe(self, client, userdata, mid):
        print("On unSubscribed: qos = %d" % mid)

    def on_publish(self, client, userdata, mid):
        print("On onPublish: qos = %d" % mid)

    def on_disconnect(self, client, userdata, rc):
        print("Unexpected disconnection rc = " + str(rc))

    def publish_message(self, topic, message):
        result = self.client.publish(topic, message)
        return result.rc  # 返回发布结果代码
    def start(self):
        # 启动 MQTT 客户端循环
        self.client.loop_start()


if __name__ == '__main__':
    mqtt_road = MqttRoad("127.0.0.1", 1883, 600)

    try:
        while True:
            # 发布消息
            message = input("Enter message to publish (or 'exit' to quit): ")
            if message.lower() == 'exit':
                break
            mqtt_road.publish_message("mqtt11", message)
            time.sleep(1)  # 可选：间隔一段时间再发送下一个消息
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        mqtt_road.client.disconnect()  # 断开连接


def testPublic(msg,topic):
    mqtt_road.publish_message(topic, msg)
    # time.sleep(1)  # 可选：间隔一段时间再发送下一个消息
