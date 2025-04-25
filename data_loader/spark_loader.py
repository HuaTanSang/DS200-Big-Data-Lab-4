from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.ml.linalg import DenseVector
import json

class DataLoader:
    def __init__(self, sc, host='localhost', port=6100, batch_interval=2):
        self.sc = sc
        self.ssc = StreamingContext(sc, batch_interval)
        self.sql_context = SQLContext(sc)
        self.stream = self.ssc.socketTextStream(host, port)

    def get_stream(self):
        return (self.stream
                .map(lambda line: json.loads(line))
                .flatMap(lambda d: d.values())
                .map(lambda x: (DenseVector(x['features']), int(x['label'])))
               )
