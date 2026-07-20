# IndustrialNeuralNetwork.ArchestrA

Biblioteka DLL dla AVEVA/Wonderware ArchestrA / System Platform QuickScript .NET.

## Cel

Projekt generuje bibliotekę:

```text
IndustrialNeuralNetwork.ArchestrA.dll
```

Target:

```text
.NET Framework 4.8
```

Biblioteka nie używa ML.NET, TensorFlow, ONNX, Torch ani bibliotek AI. Sieć neuronowa, normalizacja, trening i predykcja są zaimplementowane w C#.

## Build

Otwórz `IndustrialNeuralNetwork.ArchestrA.sln` w Visual Studio na komputerze z .NET Framework 4.8 Developer Pack i zbuduj konfigurację Release.

Albo użyj Developer Command Prompt for Visual Studio:

```cmd
msbuild IndustrialNeuralNetwork.ArchestrA.sln /p:Configuration=Release
```

Wynikowa DLL:

```text
IndustrialNeuralNetwork.ArchestrA\bin\Release\IndustrialNeuralNetwork.ArchestrA.dll
```

## Import w ArchestrA

1. Zbuduj DLL w Release.
2. W IDE ArchestrA / Application Server zaimportuj DLL jako Script Function Library.
3. Po imporcie klasa powinna być widoczna jako:

```text
IndustrialNeuralNetwork.ArchestrA.ArchestraAiService
```

## Publiczne funkcje

```csharp
string TrainModel(...)
string TrainModelAdvanced(...)
string Predict(string modelPath, string inputsCsv)
string GetModelInfo(string modelPath)
string GetLastError()
```

Metody zwracają tekst. Jeśli operacja się uda, wynik zaczyna się od `OK`. Jeśli wystąpi błąd, wynik zaczyna się od `ERROR|`.

## Wymagana procedura SQL

Biblioteka wykonuje procedurę z parametrami:

```sql
@DeviceID nvarchar(255)
@LookbackMs int
@InputCount int
@OutputCount int
```

Procedura musi zwrócić kolumny:

```text
TimeStamp
DeviceID
In_0 ... In_N
Out_0 ... Out_M
TeachingIsActive
```

Do uczenia używane są tylko rekordy, gdzie `TeachingIsActive >= 0.5`.

## Przykład QuickScript .NET: predykcja

```vbnet
dim ai as IndustrialNeuralNetwork.ArchestrA.ArchestraAiService;
dim result as System.String;
dim modelPath as System.String;
dim inputs as System.String;

ai = new IndustrialNeuralNetwork.ArchestrA.ArchestraAiService;

modelPath = "C:\Models\Generic_PM_TOX59_AM01_PUMP22115_16x5.model";
inputs = "22.6000003814697;1;22.6000003814697;0;0;22.6000003814697;228.597595214844;0;0;4.14161491394043;993.632446289063;0;0;1;1;1";

result = ai.Predict(modelPath, inputs);
LogMessage(result);
```

## Przykład QuickScript .NET: trening

Nie zaleca się uruchamiania ciężkiego treningu cyklicznie w obiekcie. Uruchamiaj trening ręcznie, zdarzeniowo albo w oknie serwisowym.

```vbnet
dim ai as IndustrialNeuralNetwork.ArchestrA.ArchestraAiService;
dim result as System.String;

ai = new IndustrialNeuralNetwork.ArchestrA.ArchestraAiService;

result = ai.TrainModel(
    "Server=SQLSERVER;Database=Runtime;Trusted_Connection=True;TrustServerCertificate=True;",
    "dbo.pm_GetDeviceData",
    "PM_TOX59_AM01_PUMP22115",
    604800000,
    16,
    5,
    0.95,
    3000,
    0.0005,
    128,
    "128;64;32",
    42,
    "C:\Models"
);

LogMessage(result);
```

## Format wejść do Predict

`Predict` przyjmuje wartości wejściowe jako tekst CSV rozdzielony średnikiem:

```text
1;2;3;4;5
```

Liczba wartości musi być równa `InputCount` zapisanemu w modelu.

## Uwaga

Plik modelu `.model` jest zapisywany jako XML, aby uniknąć zależności od zewnętrznych bibliotek JSON w środowisku .NET Framework / ArchestrA.
