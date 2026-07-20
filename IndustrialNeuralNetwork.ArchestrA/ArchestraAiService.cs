using System;
using System.Globalization;
using System.Text;

namespace IndustrialNeuralNetwork.ArchestrA
{
    public class ArchestraAiService
    {
        private string _lastError = string.Empty;

        public string GetLastError()
        {
            return _lastError ?? string.Empty;
        }

        public string TrainModel(
            string connectionString,
            string procedureName,
            string deviceId,
            int lookbackMs,
            int inputCount,
            int outputCount,
            double targetR2,
            int maxEpochs,
            double learningRate,
            int batchSize,
            string hiddenLayersCsv,
            int seed,
            string modelDirectory)
        {
            try
            {
                _lastError = string.Empty;
                int[] hiddenLayers = Csv.ParseIntCsv(hiddenLayersCsv);
                TrainingOptions options = new TrainingOptions();
                options.ConnectionString = connectionString;
                options.ProcedureName = procedureName;
                options.DeviceID = deviceId;
                options.LookbackMs = lookbackMs;
                options.InputCount = inputCount;
                options.OutputCount = outputCount;
                options.TargetR2 = targetR2;
                options.MaxEpochs = maxEpochs;
                options.LearningRate = learningRate;
                options.BatchSize = batchSize;
                options.HiddenLayers = hiddenLayers;
                options.Seed = seed;
                options.ModelDirectory = modelDirectory;

                TrainingResult result = new Trainer().Train(options);
                return "OK" +
                    "|ModelPath=" + result.ModelPath +
                    "|DeviceID=" + result.DeviceID +
                    "|InputCount=" + result.InputCount.ToString(CultureInfo.InvariantCulture) +
                    "|OutputCount=" + result.OutputCount.ToString(CultureInfo.InvariantCulture) +
                    "|Epochs=" + result.EpochsCompleted.ToString(CultureInfo.InvariantCulture) +
                    "|TargetReached=" + result.TargetReached.ToString() +
                    "|AverageR2=" + result.Metrics.Average.R2.ToString("G17", CultureInfo.InvariantCulture) +
                    "|BestValidationLoss=" + result.BestValidationLoss.ToString("G17", CultureInfo.InvariantCulture) +
                    "|TrainingRows=" + result.TrainingRows.ToString(CultureInfo.InvariantCulture) +
                    "|ValidationRows=" + result.ValidationRows.ToString(CultureInfo.InvariantCulture);
            }
            catch (Exception ex)
            {
                _lastError = ex.ToString();
                return "ERROR|" + ex.Message;
            }
        }

        public string TrainModelAdvanced(
            string connectionString,
            string procedureName,
            string deviceId,
            int lookbackMs,
            int inputCount,
            int outputCount,
            double targetR2,
            int maxEpochs,
            double learningRate,
            int batchSize,
            string hiddenLayersCsv,
            int seed,
            string modelDirectory,
            double validationRatio,
            double l2,
            double learningRateDecay,
            int earlyStoppingPatience,
            int everyNthRow,
            int minSecondsBetweenSamples)
        {
            try
            {
                _lastError = string.Empty;
                TrainingOptions options = new TrainingOptions();
                options.ConnectionString = connectionString;
                options.ProcedureName = procedureName;
                options.DeviceID = deviceId;
                options.LookbackMs = lookbackMs;
                options.InputCount = inputCount;
                options.OutputCount = outputCount;
                options.TargetR2 = targetR2;
                options.MaxEpochs = maxEpochs;
                options.LearningRate = learningRate;
                options.BatchSize = batchSize;
                options.HiddenLayers = Csv.ParseIntCsv(hiddenLayersCsv);
                options.Seed = seed;
                options.ModelDirectory = modelDirectory;
                options.ValidationRatio = validationRatio;
                options.L2 = l2;
                options.LearningRateDecay = learningRateDecay;
                options.EarlyStoppingPatience = earlyStoppingPatience;
                options.EveryNthRow = everyNthRow;
                options.MinSecondsBetweenSamples = minSecondsBetweenSamples;

                TrainingResult result = new Trainer().Train(options);
                return "OK" +
                    "|ModelPath=" + result.ModelPath +
                    "|AverageR2=" + result.Metrics.Average.R2.ToString("G17", CultureInfo.InvariantCulture) +
                    "|Epochs=" + result.EpochsCompleted.ToString(CultureInfo.InvariantCulture) +
                    "|TargetReached=" + result.TargetReached.ToString();
            }
            catch (Exception ex)
            {
                _lastError = ex.ToString();
                return "ERROR|" + ex.Message;
            }
        }

        public string Predict(string modelPath, string inputsCsv)
        {
            try
            {
                _lastError = string.Empty;
                double[] inputs = Csv.ParseDoubleCsv(inputsCsv);
                double[] outputs = new PredictionEngine(modelPath).Predict(inputs);
                return Csv.ToDoubleCsv(outputs);
            }
            catch (Exception ex)
            {
                _lastError = ex.ToString();
                return "ERROR|" + ex.Message;
            }
        }

        public string GetModelInfo(string modelPath)
        {
            try
            {
                _lastError = string.Empty;
                ModelFile model = ModelSerializer.Load(modelPath);
                StringBuilder sb = new StringBuilder();
                sb.Append("OK");
                sb.Append("|DeviceID=").Append(model.DeviceID);
                sb.Append("|InputCount=").Append(model.InputCount.ToString(CultureInfo.InvariantCulture));
                sb.Append("|OutputCount=").Append(model.OutputCount.ToString(CultureInfo.InvariantCulture));
                sb.Append("|TrainedAtUtc=").Append(model.TrainedAtUtc.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture));
                if (model.Metrics != null && model.Metrics.Average != null)
                    sb.Append("|AverageR2=").Append(model.Metrics.Average.R2.ToString("G17", CultureInfo.InvariantCulture));
                return sb.ToString();
            }
            catch (Exception ex)
            {
                _lastError = ex.ToString();
                return "ERROR|" + ex.Message;
            }
        }
    }
}
