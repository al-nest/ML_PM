using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Xml.Serialization;

namespace IndustrialNeuralNetwork.ArchestrA
{
    public enum OptimizerType { GradientDescent, Adam }
    public enum ActivationType { ReLU, LeakyReLU, Sigmoid, Tanh, Linear }

    public class TrainingOptions
    {
        public string ConnectionString { get; set; }
        public string ProcedureName { get; set; }
        public string DeviceID { get; set; }
        public int LookbackMs { get; set; }
        public int InputCount { get; set; }
        public int OutputCount { get; set; }
        public double TargetR2 { get; set; }
        public int MaxEpochs { get; set; }
        public double LearningRate { get; set; }
        public int BatchSize { get; set; }
        public int[] HiddenLayers { get; set; }
        public int Seed { get; set; }
        public string ModelDirectory { get; set; }
        public double ValidationRatio { get; set; }
        public double L2 { get; set; }
        public double LearningRateDecay { get; set; }
        public int EarlyStoppingPatience { get; set; }
        public double LeakyReluAlpha { get; set; }
        public double DropoutRate { get; set; }
        public int EveryNthRow { get; set; }
        public int MinSecondsBetweenSamples { get; set; }
        public OptimizerType Optimizer { get; set; }
        public ActivationType HiddenActivation { get; set; }

        public TrainingOptions()
        {
            ModelDirectory = ".";
            ValidationRatio = 0.2;
            L2 = 0.0001;
            LearningRateDecay = 0.0;
            EarlyStoppingPatience = 100;
            LeakyReluAlpha = 0.01;
            DropoutRate = 0.0;
            EveryNthRow = 1;
            MinSecondsBetweenSamples = 0;
            Optimizer = OptimizerType.Adam;
            HiddenActivation = ActivationType.ReLU;
        }
    }

    public class TrainingSample
    {
        public DateTime TimeStamp { get; set; }
        public string DeviceID { get; set; }
        public double[] Inputs { get; set; }
        public double[] Targets { get; set; }
    }

    public class OutputMetrics
    {
        public string OutputName { get; set; }
        public double R2 { get; set; }
        public double RMSE { get; set; }
        public double MAE { get; set; }
        public double MAPE { get; set; }
        public double AverageError { get; set; }
        public double MaxError { get; set; }
        public OutputMetrics() { }
        public OutputMetrics(string name, double r2, double rmse, double mae, double mape, double averageError, double maxError)
        {
            OutputName = name; R2 = r2; RMSE = rmse; MAE = mae; MAPE = mape; AverageError = averageError; MaxError = maxError;
        }
    }

    public class ModelMetrics
    {
        public List<OutputMetrics> PerOutput { get; set; }
        public OutputMetrics Average { get; set; }
        public ModelMetrics() { PerOutput = new List<OutputMetrics>(); Average = new OutputMetrics(); }
    }

    public class TrainingResult
    {
        public string ModelPath { get; set; }
        public string DeviceID { get; set; }
        public int InputCount { get; set; }
        public int OutputCount { get; set; }
        public int EpochsCompleted { get; set; }
        public bool TargetReached { get; set; }
        public double BestValidationLoss { get; set; }
        public ModelMetrics Metrics { get; set; }
        public DateTime TrainedAtUtc { get; set; }
        public int TrainingRows { get; set; }
        public int ValidationRows { get; set; }
    }

    public static class Csv
    {
        public static int[] ParseIntCsv(string csv)
        {
            string[] parts = Split(csv);
            int[] values = new int[parts.Length];
            for (int i = 0; i < parts.Length; i++) values[i] = int.Parse(parts[i], CultureInfo.InvariantCulture);
            return values;
        }

        public static double[] ParseDoubleCsv(string csv)
        {
            string[] parts = Split(csv);
            double[] values = new double[parts.Length];
            for (int i = 0; i < parts.Length; i++) values[i] = ParseDouble(parts[i]);
            return values;
        }

        public static string ToDoubleCsv(double[] values)
        {
            string[] parts = new string[values.Length];
            for (int i = 0; i < values.Length; i++) parts[i] = values[i].ToString("G17", CultureInfo.InvariantCulture);
            return string.Join(";", parts);
        }

        private static string[] Split(string csv)
        {
            if (string.IsNullOrWhiteSpace(csv)) return new string[0];
            return csv.Split(new char[] { ';', ',' }, StringSplitOptions.RemoveEmptyEntries).Select(x => x.Trim()).ToArray();
        }

        private static double ParseDouble(string text)
        {
            double value;
            if (double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out value)) return value;
            if (double.TryParse(text, NumberStyles.Float, CultureInfo.CurrentCulture, out value)) return value;
            string normalized = text.Replace(',', '.');
            return double.Parse(normalized, CultureInfo.InvariantCulture);
        }
    }

    public class StandardScaler
    {
        public double[] Means { get; set; }
        public double[] StdDevs { get; set; }

        public StandardScaler() { Means = new double[0]; StdDevs = new double[0]; }

        public static StandardScaler Fit(List<double[]> rows)
        {
            if (rows.Count == 0) throw new InvalidOperationException("Cannot fit scaler on empty data.");
            int n = rows[0].Length;
            double[] means = new double[n];
            double[] stds = new double[n];
            foreach (double[] row in rows) for (int i = 0; i < n; i++) means[i] += row[i];
            for (int i = 0; i < n; i++) means[i] /= rows.Count;
            foreach (double[] row in rows) for (int i = 0; i < n; i++) stds[i] += Math.Pow(row[i] - means[i], 2.0);
            for (int i = 0; i < n; i++)
            {
                stds[i] = Math.Sqrt(stds[i] / Math.Max(1, rows.Count - 1));
                if (stds[i] < 1e-12) stds[i] = 1.0;
            }
            return new StandardScaler { Means = means, StdDevs = stds };
        }

        public double[] Transform(double[] row)
        {
            if (row.Length != Means.Length) throw new ArgumentException("Invalid vector size.");
            double[] output = new double[row.Length];
            for (int i = 0; i < row.Length; i++) output[i] = (row[i] - Means[i]) / StdDevs[i];
            return output;
        }

        public double[] InverseTransform(double[] row)
        {
            if (row.Length != Means.Length) throw new ArgumentException("Invalid vector size.");
            double[] output = new double[row.Length];
            for (int i = 0; i < row.Length; i++) output[i] = row[i] * StdDevs[i] + Means[i];
            return output;
        }
    }

    public class DenseLayer
    {
        public int InputSize { get; set; }
        public int OutputSize { get; set; }
        public ActivationType Activation { get; set; }
        public double[][] Weights { get; set; }
        public double[] Biases { get; set; }
        public double DropoutRate { get; set; }

        [XmlIgnore] private double[] _lastInput;
        [XmlIgnore] private double[] _lastZ;
        [XmlIgnore] private bool[] _dropoutMask;
        [XmlIgnore] private double[][] _gradW;
        [XmlIgnore] private double[] _gradB;
        [XmlIgnore] private double[][] _mW;
        [XmlIgnore] private double[][] _vW;
        [XmlIgnore] private double[] _mB;
        [XmlIgnore] private double[] _vB;

        public DenseLayer() { }

        public DenseLayer(int inputSize, int outputSize, ActivationType activation, Random random, string initialization, double dropoutRate)
        {
            InputSize = inputSize; OutputSize = outputSize; Activation = activation; DropoutRate = dropoutRate;
            Weights = Matrix(outputSize, inputSize); Biases = new double[outputSize];
            double scale = initialization == "he" ? Math.Sqrt(2.0 / inputSize) : Math.Sqrt(2.0 / (inputSize + outputSize));
            for (int o = 0; o < outputSize; o++) for (int i = 0; i < inputSize; i++) Weights[o][i] = NextGaussian(random) * scale;
            AllocateOptimizerState();
        }

        public void AllocateOptimizerState()
        {
            _gradW = Matrix(OutputSize, InputSize); _gradB = new double[OutputSize];
            _mW = Matrix(OutputSize, InputSize); _vW = Matrix(OutputSize, InputSize);
            _mB = new double[OutputSize]; _vB = new double[OutputSize];
        }

        public void ZeroGradients()
        {
            for (int o = 0; o < OutputSize; o++) { Array.Clear(_gradW[o], 0, _gradW[o].Length); _gradB[o] = 0.0; }
        }

        public double[] Forward(double[] input, bool training, Random random, double alpha)
        {
            _lastInput = input; _lastZ = new double[OutputSize]; _dropoutMask = new bool[OutputSize];
            double[] output = new double[OutputSize];
            for (int o = 0; o < OutputSize; o++)
            {
                double z = Biases[o];
                for (int i = 0; i < InputSize; i++) z += Weights[o][i] * input[i];
                _lastZ[o] = z;
                double a = Activate(z, Activation, alpha);
                if (training && DropoutRate > 0.0 && Activation != ActivationType.Linear)
                {
                    bool keep = random.NextDouble() >= DropoutRate;
                    _dropoutMask[o] = keep;
                    a = keep ? a / (1.0 - DropoutRate) : 0.0;
                }
                else _dropoutMask[o] = true;
                output[o] = a;
            }
            return output;
        }

        public double[] Backward(double[] dOutput, double alpha)
        {
            double[] dInput = new double[InputSize];
            for (int o = 0; o < OutputSize; o++)
            {
                double d = dOutput[o];
                if (!_dropoutMask[o]) d = 0.0;
                else if (DropoutRate > 0.0 && Activation != ActivationType.Linear) d /= (1.0 - DropoutRate);
                d *= Derivative(_lastZ[o], Activation, alpha);
                _gradB[o] += d;
                for (int i = 0; i < InputSize; i++)
                {
                    _gradW[o][i] += d * _lastInput[i];
                    dInput[i] += Weights[o][i] * d;
                }
            }
            return dInput;
        }

        public void ApplyGradients(double lr, int batchSize, double l2, OptimizerType optimizer, int step)
        {
            double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
            double b1Corr = 1.0 - Math.Pow(beta1, step);
            double b2Corr = 1.0 - Math.Pow(beta2, step);
            for (int o = 0; o < OutputSize; o++)
            {
                double gb = _gradB[o] / batchSize;
                Biases[o] -= Update(ref _mB[o], ref _vB[o], gb, lr, optimizer, b1Corr, b2Corr, eps);
                for (int i = 0; i < InputSize; i++)
                {
                    double gw = _gradW[o][i] / batchSize + l2 * Weights[o][i];
                    Weights[o][i] -= Update(ref _mW[o][i], ref _vW[o][i], gw, lr, optimizer, b1Corr, b2Corr, eps);
                }
            }
        }

        private static double Update(ref double m, ref double v, double g, double lr, OptimizerType optimizer, double b1Corr, double b2Corr, double eps)
        {
            if (optimizer == OptimizerType.GradientDescent) return lr * g;
            m = 0.9 * m + 0.1 * g;
            v = 0.999 * v + 0.001 * g * g;
            return lr * (m / b1Corr) / (Math.Sqrt(v / b2Corr) + eps);
        }

        private static double Activate(double x, ActivationType type, double alpha)
        {
            switch (type)
            {
                case ActivationType.ReLU: return x > 0.0 ? x : 0.0;
                case ActivationType.LeakyReLU: return x > 0.0 ? x : alpha * x;
                case ActivationType.Sigmoid: return 1.0 / (1.0 + Math.Exp(-Math.Max(-60.0, Math.Min(60.0, x))));
                case ActivationType.Tanh: return Math.Tanh(x);
                default: return x;
            }
        }

        private static double Derivative(double z, ActivationType type, double alpha)
        {
            switch (type)
            {
                case ActivationType.ReLU: return z > 0.0 ? 1.0 : 0.0;
                case ActivationType.LeakyReLU: return z > 0.0 ? 1.0 : alpha;
                case ActivationType.Sigmoid:
                    double s = Activate(z, ActivationType.Sigmoid, alpha); return s * (1.0 - s);
                case ActivationType.Tanh:
                    double t = Math.Tanh(z); return 1.0 - t * t;
                default: return 1.0;
            }
        }

        private static double[][] Matrix(int rows, int cols)
        {
            double[][] m = new double[rows][];
            for (int r = 0; r < rows; r++) m[r] = new double[cols];
            return m;
        }

        private static double NextGaussian(Random random)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }
    }

    public class NeuralNetwork
    {
        public List<DenseLayer> Layers { get; set; }
        public double LeakyReluAlpha { get; set; }

        public NeuralNetwork() { Layers = new List<DenseLayer>(); LeakyReluAlpha = 0.01; }

        public static NeuralNetwork Create(int inputSize, int outputSize, int[] hiddenLayers, ActivationType hiddenActivation, double dropoutRate, int seed, double alpha)
        {
            NeuralNetwork n = new NeuralNetwork();
            n.LeakyReluAlpha = alpha;
            Random random = new Random(seed);
            int prev = inputSize;
            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                string init = hiddenActivation == ActivationType.ReLU || hiddenActivation == ActivationType.LeakyReLU ? "he" : "xavier";
                n.Layers.Add(new DenseLayer(prev, hiddenLayers[i], hiddenActivation, random, init, dropoutRate));
                prev = hiddenLayers[i];
            }
            n.Layers.Add(new DenseLayer(prev, outputSize, ActivationType.Linear, random, "xavier", 0.0));
            return n;
        }

        public void EnsureOptimizerState()
        {
            foreach (DenseLayer layer in Layers) layer.AllocateOptimizerState();
        }

        public double[] Predict(double[] input)
        {
            double[] a = input;
            Random r = new Random(0);
            foreach (DenseLayer layer in Layers) a = layer.Forward(a, false, r, LeakyReluAlpha);
            return a;
        }

        public double[] Forward(double[] input, bool training, Random random)
        {
            double[] a = input;
            foreach (DenseLayer layer in Layers) a = layer.Forward(a, training, random, LeakyReluAlpha);
            return a;
        }

        public void Backward(double[] predicted, double[] target)
        {
            double[] grad = new double[predicted.Length];
            for (int i = 0; i < grad.Length; i++) grad[i] = 2.0 * (predicted[i] - target[i]) / grad.Length;
            for (int i = Layers.Count - 1; i >= 0; i--) grad = Layers[i].Backward(grad, LeakyReluAlpha);
        }

        public void ZeroGradients()
        {
            foreach (DenseLayer layer in Layers) layer.ZeroGradients();
        }

        public void ApplyGradients(double lr, int batchSize, double l2, OptimizerType optimizer, int step)
        {
            foreach (DenseLayer layer in Layers) layer.ApplyGradients(lr, batchSize, l2, optimizer, step);
        }
    }

    public class ModelFile
    {
        public string Format { get; set; }
        public DateTime TrainedAtUtc { get; set; }
        public string DeviceID { get; set; }
        public int InputCount { get; set; }
        public int OutputCount { get; set; }
        public int[] Architecture { get; set; }
        public double LearningRate { get; set; }
        public double L2 { get; set; }
        public StandardScaler InputScaler { get; set; }
        public StandardScaler OutputScaler { get; set; }
        public NeuralNetwork Network { get; set; }
        public ModelMetrics Metrics { get; set; }
        public ModelFile() { Format = "IndustrialNeuralNetwork.ArchestrA.Model.v1"; }
    }

    public static class ModelSerializer
    {
        public static void Save(ModelFile model, string path)
        {
            XmlSerializer serializer = new XmlSerializer(typeof(ModelFile));
            using (FileStream fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None)) serializer.Serialize(fs, model);
        }

        public static ModelFile Load(string path)
        {
            XmlSerializer serializer = new XmlSerializer(typeof(ModelFile));
            using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                ModelFile model = (ModelFile)serializer.Deserialize(fs);
                model.Network.EnsureOptimizerState();
                return model;
            }
        }
    }

    public class StoredProcedureTrainingDataProvider
    {
        public List<TrainingSample> Load(TrainingOptions options)
        {
            List<TrainingSample> rows = new List<TrainingSample>();
            using (SqlConnection connection = new SqlConnection(options.ConnectionString))
            {
                connection.Open();
                using (SqlCommand command = new SqlCommand(options.ProcedureName, connection))
                {
                    command.CommandType = CommandType.StoredProcedure;
                    command.CommandTimeout = 300;
                    command.Parameters.AddWithValue("@DeviceID", options.DeviceID);
                    command.Parameters.AddWithValue("@LookbackMs", options.LookbackMs);
                    command.Parameters.AddWithValue("@InputCount", options.InputCount);
                    command.Parameters.AddWithValue("@OutputCount", options.OutputCount);
                    using (SqlDataReader reader = command.ExecuteReader())
                    {
                        int rowIndex = 0;
                        DateTime? lastAccepted = null;
                        while (reader.Read())
                        {
                            rowIndex++;
                            if (options.EveryNthRow > 1 && rowIndex % options.EveryNthRow != 0) continue;
                            double teaching = Convert.ToDouble(reader["TeachingIsActive"], CultureInfo.InvariantCulture);
                            if (teaching < 0.5) continue;
                            DateTime ts = Convert.ToDateTime(reader["TimeStamp"], CultureInfo.InvariantCulture);
                            if (options.MinSecondsBetweenSamples > 0 && lastAccepted.HasValue && (ts - lastAccepted.Value).TotalSeconds < options.MinSecondsBetweenSamples) continue;
                            double[] x = ReadVector(reader, "In_", options.InputCount);
                            double[] y = ReadVector(reader, "Out_", options.OutputCount);
                            if (AllFinite(x) && AllFinite(y))
                            {
                                TrainingSample s = new TrainingSample();
                                s.TimeStamp = ts;
                                s.DeviceID = Convert.ToString(reader["DeviceID"], CultureInfo.InvariantCulture);
                                s.Inputs = x;
                                s.Targets = y;
                                rows.Add(s);
                                lastAccepted = ts;
                            }
                        }
                    }
                }
            }
            return rows;
        }

        private static double[] ReadVector(SqlDataReader reader, string prefix, int count)
        {
            double[] v = new double[count];
            for (int i = 0; i < count; i++)
            {
                object raw = reader[prefix + i.ToString(CultureInfo.InvariantCulture)];
                v[i] = raw == DBNull.Value ? double.NaN : Convert.ToDouble(raw, CultureInfo.InvariantCulture);
            }
            return v;
        }

        private static bool AllFinite(double[] values)
        {
            for (int i = 0; i < values.Length; i++) if (double.IsNaN(values[i]) || double.IsInfinity(values[i])) return false;
            return true;
        }
    }

    public static class MetricsCalculator
    {
        public static ModelMetrics Calculate(List<double[]> actual, List<double[]> predicted, int outputCount)
        {
            ModelMetrics mm = new ModelMetrics();
            for (int o = 0; o < outputCount; o++)
            {
                double[] a = actual.Select(v => v[o]).ToArray();
                double[] p = predicted.Select(v => v[o]).ToArray();
                mm.PerOutput.Add(Calc("Out_" + o.ToString(CultureInfo.InvariantCulture), a, p));
            }
            mm.Average = new OutputMetrics(
                "Average",
                mm.PerOutput.Average(m => m.R2),
                mm.PerOutput.Average(m => m.RMSE),
                mm.PerOutput.Average(m => m.MAE),
                mm.PerOutput.Average(m => m.MAPE),
                mm.PerOutput.Average(m => m.AverageError),
                mm.PerOutput.Average(m => m.MaxError));
            return mm;
        }

        private static OutputMetrics Calc(string name, double[] actual, double[] predicted)
        {
            int n = actual.Length;
            double mean = actual.Average();
            double ssTot = actual.Sum(x => Math.Pow(x - mean, 2.0));
            double ssRes = 0.0;
            double sumAbs = 0.0;
            double sumSigned = 0.0;
            double maxAbs = 0.0;
            double sumApe = 0.0;
            int apeCount = 0;
            for (int i = 0; i < n; i++)
            {
                double e = predicted[i] - actual[i];
                double ae = Math.Abs(e);
                ssRes += e * e;
                sumAbs += ae;
                sumSigned += e;
                if (ae > maxAbs) maxAbs = ae;
                if (Math.Abs(actual[i]) > 1e-12) { sumApe += Math.Abs(e / actual[i]) * 100.0; apeCount++; }
            }
            double r2 = ssTot <= 1e-12 ? 1.0 : 1.0 - ssRes / ssTot;
            return new OutputMetrics(name, r2, Math.Sqrt(ssRes / n), sumAbs / n, apeCount == 0 ? 0.0 : sumApe / apeCount, sumSigned / n, maxAbs);
        }
    }

    public class Trainer
    {
        public TrainingResult Train(TrainingOptions options)
        {
            Validate(options);
            if (!Directory.Exists(options.ModelDirectory)) Directory.CreateDirectory(options.ModelDirectory);
            List<TrainingSample> samples = new StoredProcedureTrainingDataProvider().Load(options);
            if (samples.Count < 20) throw new InvalidOperationException("At least 20 TeachingIsActive records are required.");
            Random random = new Random(options.Seed);
            Shuffle(samples, random);
            int valCount = Math.Max(1, (int)Math.Round(samples.Count * options.ValidationRatio));
            List<TrainingSample> validation = samples.Take(valCount).ToList();
            List<TrainingSample> training = samples.Skip(valCount).ToList();
            StandardScaler inputScaler = StandardScaler.Fit(training.Select(s => s.Inputs).ToList());
            StandardScaler outputScaler = StandardScaler.Fit(training.Select(s => s.Targets).ToList());
            List<double[]> xTrain = training.Select(s => inputScaler.Transform(s.Inputs)).ToList();
            List<double[]> yTrain = training.Select(s => outputScaler.Transform(s.Targets)).ToList();
            NeuralNetwork net = NeuralNetwork.Create(options.InputCount, options.OutputCount, options.HiddenLayers, options.HiddenActivation, options.DropoutRate, options.Seed, options.LeakyReluAlpha);
            NeuralNetwork best = Clone(net);
            ModelMetrics bestMetrics = Evaluate(best, inputScaler, outputScaler, validation, options.OutputCount);
            double bestLoss = Loss(best, inputScaler, outputScaler, validation, options.OutputCount);
            bool targetReached = bestMetrics.Average.R2 >= options.TargetR2;
            int noImprove = 0;
            int completed = 0;
            int step = 0;
            for (int epoch = 1; epoch <= options.MaxEpochs; epoch++)
            {
                completed = epoch;
                int[] order = Enumerable.Range(0, xTrain.Count).ToArray();
                Shuffle(order, random);
                double lr = options.LearningRate / (1.0 + options.LearningRateDecay * (epoch - 1));
                for (int start = 0; start < order.Length; start += options.BatchSize)
                {
                    int bs = Math.Min(options.BatchSize, order.Length - start);
                    net.ZeroGradients();
                    for (int b = 0; b < bs; b++)
                    {
                        int idx = order[start + b];
                        double[] pred = net.Forward(xTrain[idx], true, random);
                        net.Backward(pred, yTrain[idx]);
                    }
                    step++;
                    net.ApplyGradients(lr, bs, options.L2, options.Optimizer, step);
                }
                ModelMetrics m = Evaluate(net, inputScaler, outputScaler, validation, options.OutputCount);
                double loss = Loss(net, inputScaler, outputScaler, validation, options.OutputCount);
                if (loss + 1e-12 < bestLoss)
                {
                    bestLoss = loss; bestMetrics = m; best = Clone(net); noImprove = 0;
                }
                else noImprove++;
                if (bestMetrics.Average.R2 >= options.TargetR2) { targetReached = true; break; }
                if (options.EarlyStoppingPatience > 0 && noImprove >= options.EarlyStoppingPatience) break;
            }
            ModelFile model = new ModelFile();
            model.TrainedAtUtc = DateTime.UtcNow;
            model.DeviceID = options.DeviceID;
            model.InputCount = options.InputCount;
            model.OutputCount = options.OutputCount;
            model.Architecture = new int[] { options.InputCount }.Concat(options.HiddenLayers).Concat(new int[] { options.OutputCount }).ToArray();
            model.LearningRate = options.LearningRate;
            model.L2 = options.L2;
            model.InputScaler = inputScaler;
            model.OutputScaler = outputScaler;
            model.Network = best;
            model.Metrics = bestMetrics;
            string path = Path.Combine(options.ModelDirectory, "Generic_" + Safe(options.DeviceID) + "_" + options.InputCount + "x" + options.OutputCount + ".model");
            ModelSerializer.Save(model, path);
            TrainingResult result = new TrainingResult();
            result.ModelPath = path; result.DeviceID = options.DeviceID; result.InputCount = options.InputCount; result.OutputCount = options.OutputCount;
            result.EpochsCompleted = completed; result.TargetReached = targetReached; result.BestValidationLoss = bestLoss; result.Metrics = bestMetrics;
            result.TrainedAtUtc = model.TrainedAtUtc; result.TrainingRows = training.Count; result.ValidationRows = validation.Count;
            return result;
        }

        private static void Validate(TrainingOptions o)
        {
            if (string.IsNullOrWhiteSpace(o.ConnectionString)) throw new ArgumentException("ConnectionString is required.");
            if (string.IsNullOrWhiteSpace(o.ProcedureName)) throw new ArgumentException("ProcedureName is required.");
            if (string.IsNullOrWhiteSpace(o.DeviceID)) throw new ArgumentException("DeviceID is required.");
            if (o.LookbackMs <= 0 || o.InputCount <= 0 || o.OutputCount <= 0 || o.MaxEpochs <= 0 || o.LearningRate <= 0.0 || o.BatchSize <= 0) throw new ArgumentException("Invalid training parameters.");
            if (o.HiddenLayers == null || o.HiddenLayers.Length == 0) throw new ArgumentException("HiddenLayers are required.");
        }

        private static ModelMetrics Evaluate(NeuralNetwork net, StandardScaler xs, StandardScaler ys, List<TrainingSample> samples, int outCount)
        {
            List<double[]> actual = new List<double[]>();
            List<double[]> predicted = new List<double[]>();
            foreach (TrainingSample s in samples)
            {
                predicted.Add(ys.InverseTransform(net.Predict(xs.Transform(s.Inputs))));
                actual.Add(s.Targets);
            }
            return MetricsCalculator.Calculate(actual, predicted, outCount);
        }

        private static double Loss(NeuralNetwork net, StandardScaler xs, StandardScaler ys, List<TrainingSample> samples, int outCount)
        {
            double loss = 0.0;
            foreach (TrainingSample s in samples)
            {
                double[] p = net.Predict(xs.Transform(s.Inputs));
                double[] y = ys.Transform(s.Targets);
                for (int i = 0; i < outCount; i++) loss += Math.Pow(p[i] - y[i], 2.0);
            }
            return loss / (samples.Count * outCount);
        }

        private static NeuralNetwork Clone(NeuralNetwork src)
        {
            NeuralNetwork c = new NeuralNetwork(); c.LeakyReluAlpha = src.LeakyReluAlpha;
            foreach (DenseLayer l in src.Layers)
            {
                DenseLayer d = new DenseLayer();
                d.InputSize = l.InputSize; d.OutputSize = l.OutputSize; d.Activation = l.Activation; d.DropoutRate = l.DropoutRate;
                d.Biases = (double[])l.Biases.Clone();
                d.Weights = new double[l.Weights.Length][];
                for (int r = 0; r < l.Weights.Length; r++) d.Weights[r] = (double[])l.Weights[r].Clone();
                d.AllocateOptimizerState();
                c.Layers.Add(d);
            }
            return c;
        }

        private static void Shuffle<T>(IList<T> values, Random random)
        {
            for (int i = values.Count - 1; i > 0; i--) { int j = random.Next(i + 1); T tmp = values[i]; values[i] = values[j]; values[j] = tmp; }
        }

        private static string Safe(string value)
        {
            char[] chars = value.Select(ch => char.IsLetterOrDigit(ch) ? ch : '_').ToArray();
            return new string(chars);
        }
    }

    public class PredictionEngine
    {
        private readonly ModelFile _model;
        public PredictionEngine(string modelPath) { _model = ModelSerializer.Load(modelPath); }
        public double[] Predict(double[] inputs)
        {
            if (inputs.Length != _model.InputCount) throw new ArgumentException("Input vector has invalid size. Expected " + _model.InputCount.ToString(CultureInfo.InvariantCulture));
            return _model.OutputScaler.InverseTransform(_model.Network.Predict(_model.InputScaler.Transform(inputs)));
        }
    }
}
