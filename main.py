from pyspark import SparkContext
from data_loader.spark_loader import DataLoader
from model.lenet import LeNet
from trainer.trainer import Trainer

if __name__ == "__main__":
    sc = SparkContext(appName="FashionMNIST_Streaming_LeNet")
    loader = DataLoader(sc, host='localhost', port=6100, batch_interval=2)
    model = LeNet()
    trainer = Trainer(model, loader.sql_context)

    stream = loader.get_stream()
    stream.foreachRDD(lambda t, rdd: trainer.train(rdd))
    loader.ssc.start()
    loader.ssc.awaitTermination()
