import requests
import time


def create_kafka_postISRmedian_topics():
    """This function creates the kafka topics for postISR pixel counts.
    Note: I do not understand if you're supposed to only do this once
    or do it every time a new cluster is made, or something else.
    Thus I don't really know how this function should be written or if it
    should be a function at all."""

    # Note: I'm not sure if you want this on usdf or the summit
    # or the option of doing both
    sasquatch_rest_proxy_urls = [
        "https://summit-lsp.lsst.codes/sasquatch-rest-proxy",
        "https://usdf-rsp-dev.slac.stanford.edu/sasquatch-rest-proxy",
    ]

    headers = {"content-type": "application/json"}

    # make a list of the topics you want to create
    # I have no idea what reasonable partition counts or replication
    # factors are, so I just copied from the tutorial
    # Not sure of the correct topic name either
    all_topic_configs = [
        {
            "topic_name": "lsst.dm.latiss.postIsrPixelMedian",
            "partitions_count": 1,
            "replication_factor": 3,
        },
        {
            "topic_name": "lsst.dm.comcam.postIsrPixelMedian",
            "partitions_count": 1,
            "replication_factor": 3,
        },
    ]

    for sasquatch_url in sasquatch_rest_proxy_urls:
        # get cluster id
        r = requests.get(f"{sasquatch_url}/v3/clusters", headers=headers)

        cluster_id = r.json()["data"][0]["cluster_id"]

        headers = {"content-type": "application/json"}

        # create your kafka topics
        for topic_config in all_topic_configs:
            response = requests.post(
                f"{sasquatch_url}/v3/clusters/{cluster_id}/topics",
                json=topic_config,
                headers=headers,
            )

        print(response.text)  # yes I know this is terrible and I should use a logger


def post_to_sasquatch_latiss_isr(timestamp, obsid, postIsrPixelMedian):
    """I think this function posts to sasquatch"""

    # not sure again if this will be summit or usdf
    url = (
        "https://usdf-rsp-dev.slac.stanford.edu/sasquatch-rest-proxy/topics/lsst.dm.latiss.postIsrPixelMedian"
    )

    payload = {
        "value_schema": '{"namespace": "lsst.dm.latiss", "type": "record", \
            "name": "postIsrPixelMedian", "fields": \
                [{"name": "timestamp", "type": "long"}, \
                    {"name": "obsid", "type": "integer"}, \
                        {"name": "instrument", "type": "string", "default": "LATISS"}, \
                            {"name": "postIsrPixelMedian","type": "float"}]}',
        "records": [
            {
                "value": {
                    "timestamp": timestamp,
                    "obsid": obsid,
                    "instrument": "LATISS",
                    "postIsrPixelMedian": postIsrPixelMedian,
                }
            }
        ],
    }
    headers = {
        "Content-Type": "application/vnd.kafka.avro.v2+json",
        "Accept": "application/vnd.kafka.v2+json",
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    print(response.text)


def post_to_sasquatch_comcam_isr(
    timestamp,
    obsid,
    postIsrPixelMedian,
    postIsrPixelMedianMedian,
    postIsrPixelMedianMean,
    postIsrPixelMedianMax,
):
    """I think this function posts to sasquatch"""

    # not sure again if this will be summit or usdf
    url = (
        "https://usdf-rsp-dev.slac.stanford.edu/sasquatch-rest-proxy/topics/lsst.dm.latiss.postIsrPixelMedian"
    )

    payload = {
        "value_schema": '{"namespace": "lsst.dm.comcam", "type": "record", \
            "name": "postIsrPixelMedian", "fields": [{"name": "timestamp", \
                "type": "long"}, {"name": "obsid", "type": "integer"}, \
                    {"name": "instrument", "type": "string", "default": "ComCam"}, \
                        {"name": "postIsrPixelMedian","type": "float"},\
                            {"name": "postIsrPixelMedianMedian","type": "float"}, \
                                {"name": "postIsrPixelMedianMean","type": "float"}, \
                                    {"name": "postIsrPixelMedianMax","type": "float"}]}',
        "records": [
            {
                "value": {
                    "timestamp": timestamp,
                    "obsid": obsid,
                    "instrument": "ComCam",
                    "postIsrPixelMedian": postIsrPixelMedian,
                    "postIsrPixelMedianMedian": postIsrPixelMedianMedian,
                    "postIsrPixelMedianMean": postIsrPixelMedianMean,
                    "postIsrPixelMedianMax": postIsrPixelMedianMax,
                }
            }
        ],
    }
    headers = {
        "Content-Type": "application/vnd.kafka.avro.v2+json",
        "Accept": "application/vnd.kafka.v2+json",
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    print(response.text)


""" Making the listener"""


def listen_to_kafka(topic, key, obsid, broker):
    """
    topic is Topic name, e.g. 'lsst.dm.latiss.postIsrPixelMedian'
    key is key e.g. postIsrPixelMedian
    """

    # not sure again if this will be summit or usdf
    url = f"https://usdf-rsp-dev.slac.stanford.edu/sasquatch-rest-proxy/topics/{topic}"

    headers = {
        "Content-Type": "application/vnd.kafka.avro.v2+json",
        "Accept": "application/vnd.kafka.v2+json",
    }

    while True:  # probably replace with a timeout of some kind, and better polling
        response = requests.request("GET", url, headers=headers)
        # need to check whether these are the right outputs
        if response["value"]["obsid"] == obsid:
            return response["value"]["key"]

        time.sleep(0.5)
