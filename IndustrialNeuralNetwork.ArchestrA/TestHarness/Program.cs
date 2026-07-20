using System;
using System.IO;
using IndustrialNeuralNetwork.ArchestrA;

namespace TestHarness
{
    internal static class Program
    {
        private static void Main(string[] args)
        {
            Console.WriteLine("Start testu DLL IndustrialNeuralNetwork.ArchestrA");

            ArchestraAiService ai = new ArchestraAiService();

            Console.WriteLine("Obiekt ArchestraAiService utworzony poprawnie.");
            Console.WriteLine("LastError: " + ai.GetLastError());

            string modelPath = @"C:\Models\Generic_PM_TOX59_AM01_PUMP22115_16x5.model";

            if (File.Exists(modelPath))
            {
                Console.WriteLine("Znaleziono model: " + modelPath);

                string inputs =
                    "22.6000003814697;" +
                    "1;" +
                    "22.6000003814697;" +
                    "0;" +
                    "0;" +
                    "22.6000003814697;" +
                    "228.597595214844;" +
                    "0;" +
                    "0;" +
                    "4.14161491394043;" +
                    "993.632446289063;" +
                    "0;" +
                    "0;" +
                    "1;" +
                    "1;" +
                    "1";

                string prediction = ai.Predict(modelPath, inputs);

                Console.WriteLine("Wynik predykcji:");
                Console.WriteLine(prediction);
            }
            else
            {
                Console.WriteLine("Nie znaleziono modelu:");
                Console.WriteLine(modelPath);
                Console.WriteLine("Test ładowania DLL zakończony poprawnie, ale predykcji nie wykonano.");
            }

            Console.WriteLine("Koniec testu.");
            Console.ReadLine();
        }
    }
}