/*-- 
    Copyright (c) 2017, 레드니체|redniche|Redniche|redniche@naver.com
 
    This file is licenced under a Creative Commons license: 
    http://creativecommons.org/licenses/by/2.5/ 
--*/

using System;
using System.IO;
using System.Windows.Forms;
using System.Drawing;
using System.Runtime.Serialization.Formatters.Binary;

namespace 신경망프로젝트
{

    [Serializable]
    class NeuralNetwork
    {
        //입력 은닉 출력 노드수와 학습률 멤버
        private Int32 inode;
        private Int32 hnode;
        private Int32 onode;
        private Double lrate;
        private Int32 count;
        //가중치 배열에 대한 멤버
        private Double[,] wih;
        private Double[,] who;

        //생성자: 각멤버들을 초기화합니다.
        public NeuralNetwork(Int32 inputnodes, Int32 hiddennodes, Int32 outputnodes, Double learningrate)
        {
            inode = inputnodes;
            hnode = hiddennodes;
            onode = outputnodes;
            lrate = learningrate;

            count = 0; //학습 횟수

            wih = new Double[inode, hnode];
            who = new Double[hnode, onode];

            Random rand = new Random();

            for (Int32 j = 0; j < hnode; j++)
            {
                for (Int32 i = 0; i < hnode; i++)
                    wih[i, j] = rand.NextDouble() - 0.5;
            }
            for (Int32 j = 0; j < onode; j++)
            {
                for (Int32 i = 0; i < hnode; i++)
                    who[i, j] = rand.NextDouble() - 0.5;
            }
        }

        public void Train(Double[] inputs_list, Double[] targets_list)
        {
            #region 정전파
            Double[] hidden_inputs = new Double[hnode];

            for (Int32 j = 0; j < hnode; j++)
            {
                for (Int32 i = 0; i < inode; i++)
                    hidden_inputs[j] += wih[i, j] * inputs_list[i];
            }
            Double[] hidden_outputs = Calculate.Activation_function(hidden_inputs);

            Double[] final_inputs = new Double[onode];

            for (Int32 j = 0; j < onode; j++)
            {
                for (Int32 i = 0; i < hnode; i++)
                    final_inputs[j] += who[i, j] * hidden_outputs[i];
            }
            Double[] final_outputs = Calculate.Activation_function(final_inputs);

            #endregion

            #region 역전파

            Double[] output_errors = targets_list;
            for (Int32 i = 0; i < onode; i++)
                output_errors[i] -= final_outputs[i];

            Double[] hidden_errors = new Double[hnode];
            for (Int32 j = 0; j < hnode; j++)
                for (Int32 i = 0; i < onode; i++)
                    hidden_errors[j] += who[j, i] * output_errors[i];

            //------------------가중치계산(미분이 포함된 가중치 업데이트)
            for (Int32 j = 0; j < onode; j++)
                for (Int32 i = 0; i < hnode; i++)
                    who[i, j] += lrate * output_errors[j] * final_outputs[j] * (1d - final_outputs[j]) * hidden_outputs[i];

            for (Int32 j = 0; j < hnode; j++)
                for (Int32 i = 0; i < inode; i++)
                    wih[i, j] += lrate * hidden_errors[j] * hidden_outputs[j] * (1d - hidden_outputs[j]) * inputs_list[i];
            //------------------가중치계산 끝
            #endregion
            count++;
        }
        public Double[] Query(Double[] inputs_list)
        {
            #region 정전파
            Double[] hidden_inputs = new Double[hnode];

            for (Int32 j = 0; j < hnode; j++)
            {
                for (Int32 i = 0; i < inode; i++)
                    hidden_inputs[j] += wih[i, j] * inputs_list[i];
            }
            Double[] hidden_outputs = Calculate.Activation_function(hidden_inputs);

            Double[] final_inputs = new Double[onode];

            for (Int32 j = 0; j < onode; j++)
            {
                for (Int32 i = 0; i < hnode; i++)
                    final_inputs[j] += who[i, j] * hidden_outputs[i];
            }
            return Calculate.Activation_function(final_inputs);
            #endregion
        }
        public void Show_Status()
        {
            Console.WriteLine("┌──────────신경망정보───────────┐");
            Console.WriteLine("│ 입력층 노드: " + inode + "개(" + Math.Sqrt(inode) + "px)\t│");
            Console.WriteLine("│ 은닉층 노드: " + hnode + "개\t\t│");
            Console.WriteLine("│ 출력층 노드: " + onode + "개\t\t│");
            Console.WriteLine("│ 학습률: " + lrate + "개\t\t\t│");
            Console.WriteLine("│ 학습된 횟수: " + count + "개\t\t│");
            Console.WriteLine("└───────────────────────────────┘");
        }
    }
    static class Calculate
    {
        public const Double E = 2.7182818284590452353602874;
        public static Double[] Activation_function(Double[] arr)
        {
            Double[] result = new Double[arr.Length];
            for (Int32 i = 0; i < result.Length; i++)
                result[i] = 1d / (1d + Math.Pow(E, -arr[i]));
            return result;
        }
    }
    static class FileIO
    {
        public static String CSV_Open()
        {
            using (OpenFileDialog dlgOpen = new OpenFileDialog())
            {
                dlgOpen.Filter = "CSV File|*.csv";
                dlgOpen.Title = "CSV파일 열기";
                if (dlgOpen.ShowDialog() == DialogResult.OK)
                {
                    try
                    {
                        return dlgOpen.FileName;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("오류발생:" + ex);
                        return null;
                    }
                }
            }
            return null;
        }
        public static Double[][] CSV_Read(String path)
        {
            try
            {
                String[] strs = File.ReadAllLines(path);
                Int32 len = strs.Length;
                Double[][] tmp = new Double[len][];
                for (Int32 i = 0; i < len; i++)
                    tmp[i] = Array.ConvertAll(strs[i].Split(','), Convert.ToDouble);
                foreach (Double[] t in tmp)
                    for (Int32 i = 1; i < t.Length; i++)
                        t[i] = t[i] / 255 + 0.01;
                return tmp;
            }
            catch (Exception ex)
            {
                Console.WriteLine("오류발생:" + ex);
                return new Double[0][];
            }
        }
        public static String[] Image_Open()
        {
            using (OpenFileDialog dlgOpen = new OpenFileDialog())
            {
                dlgOpen.Filter = "IMAGE File|*.png;*.jpg;*.jpeg";
                dlgOpen.Title = "이미지파일 열기";
                dlgOpen.Multiselect = true;
                if (dlgOpen.ShowDialog() == DialogResult.OK)
                {
                    try
                    {
                        return dlgOpen.FileNames;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("오류");
                        return null;
                    }
                }
            }
            return null;
        }
        public static Double[] Image_Read(String filename)
        {
            using (Bitmap bmp = new Bitmap(filename))
            {
                Double[] matrix = new Double[bmp.Width * bmp.Height + 1];
                Int32 sub = filename.LastIndexOf('\\') + 1;
                try { matrix[0] = Int32.Parse(filename.Substring(sub, 1)); }
                catch (Exception ex)
                {
                    matrix[0] = -1;
                }
                Int32 i = 1;
                for (Int32 y = 0; y < bmp.Height; y++)
                    for (Int32 x = 0; x < bmp.Width; x++)
                        matrix[i++] = (1.0 - (Double)bmp.GetPixel(x, y).GetBrightness()) * 0.99 + 0.01;
                return matrix;
            }
        }
        public static String Neural_Open()
        {
            using (OpenFileDialog dlgOpen = new OpenFileDialog())
            {
                dlgOpen.Filter = "neu File|*.neu";
                dlgOpen.Title = "학습된 신경망 파일 열기";
                if (dlgOpen.ShowDialog() == DialogResult.OK)
                {
                    try
                    {
                        return dlgOpen.FileName;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("오류: " + ex);
                        return null;
                    }
                }
            }
            return null;
        }
        public static string Neural_Save()
        {
            using (SaveFileDialog dlgOpen = new SaveFileDialog())
            {
                dlgOpen.Filter = "neu File|*.neu";
                dlgOpen.Title = "학습된 신경망 파일 저장하기";
                if (dlgOpen.ShowDialog() == DialogResult.OK)
                {
                    try
                    {
                        return dlgOpen.FileName;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("오류: " + ex);
                        return null;
                    }
                }
            }
            return null;
        }
        public static void CSV_Change()
        {
            String[] change_tmp = FileIO.Image_Open();
            String result = "";
            foreach (String tmp in change_tmp)
                using (Bitmap bmp = new Bitmap(tmp))
                {
                    Double[] matrix = new Double[bmp.Width * bmp.Height + 1];
                    Int32 sub = tmp.LastIndexOf('\\') + 1;
                    result += tmp.Substring(sub, 1) + ",";
                    Int32 h = bmp.Height;
                    Int32 w = bmp.Width;
                    for (Int32 y = 0; y < h; y++)
                        for (Int32 x = 0; x < w; x++)
                        {
                            result += ((1f - bmp.GetPixel(x, y).GetBrightness()) * 255).ToString();
                            if (!(y == h - 1 && x == w - 1))
                                result += ",";
                        }
                    result += "\n";
                }
            String path = "";
            using (SaveFileDialog dlgOpen = new SaveFileDialog())
            {
                dlgOpen.Filter = "CSV File|*.csv";
                dlgOpen.Title = "CSV파일 저장하기";
                if (dlgOpen.ShowDialog() == DialogResult.OK)
                {
                    try
                    {
                        path = dlgOpen.FileName;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("오류: " + ex);
                        return;
                    }
                }
            }
            StreamWriter sw = new StreamWriter(path);
            sw.Write(result);
            sw.Close();
        }
    }


    class Program
    {

        //NeualNetwork 클래스와 종속성을 가지고 있는 클래스입니다.

        static void MI_Train(NeuralNetwork n, Double[] inputs)
        {
            Int32 label = (Int32)inputs[0];
            Int32 l = inputs.Length - 1;
            Double[] real_input = new Double[l];

            Array.Copy(inputs, 1, real_input, 0, l);

            Double[] targets = { 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01 };
            targets[label] = 0.99;
            n.Train(real_input, targets);
        }
        static bool MI_Query(NeuralNetwork n, Double[] inputs)
        {
            Int32 label = (Int32)inputs[0];
            Int32 l = inputs.Length - 1;
            Double[] real_input = new Double[l];

            Array.Copy(inputs, 1, real_input, 0, l);
            Double[] answer = n.Query(real_input);

            Double max = 0;
            Console.Write("질의결과: ");
            foreach (Double d in answer)
                if (d > max)
                    max = d;
            Console.WriteLine("정답:{0} 대답:{1}", label, Array.IndexOf(answer, max));
            if (Array.IndexOf(answer, max) == label)
                return true;
            return false;
        }
        [STAThread]
        static void Main()
        {
            Int32 input_node;
            Int32 hidden_nodes;
            Int32 output_nodes;
            Double learning_rate;

            NeuralNetwork n = null;

            String selection;
            Console.WriteLine("명령어 사용법을 보시려면 \"Help\" 입력하십시오.");
            while (n == null)
            {
                Console.Write(">");
                selection = Console.ReadLine();
                if (selection == "Create")
                {
                    Console.WriteLine("신경망 클래스를 생성합니다.");
                    Console.Write("입력층 노드 수(pixel 수):");
                    input_node = Int32.Parse(Console.ReadLine());
                    Console.Write("은닉층 노드 수:");
                    hidden_nodes = Int32.Parse(Console.ReadLine());
                    Console.Write("출력층 노드 수(숫자수: 10):");
                    output_nodes = Int32.Parse(Console.ReadLine());
                    Console.Write("학습률(실수):");
                    learning_rate = Double.Parse(Console.ReadLine());

                    n = new NeuralNetwork(input_node, hidden_nodes, output_nodes, learning_rate);

                    Console.WriteLine("생성되었습니다.");
                }
                if (selection == "Help")
                {
                    Console.WriteLine(@"명령어를 알려드립니다.
//신경망 객체가 없습니다.
Help: 이 출력창을 콘솔창에 출력합니다.
Create: 새로운 신경망 객체를 생성합니다.
Open: 원래있던 학습된 신경망을 불러옵니다.
CsvChange: 이미지 파일을 CSV파일로 저장합니다.
");
                }
                if (selection == "Open")
                {
                    try
                    {
                        String s;
                        if ((s = FileIO.Neural_Open()) != null)
                        {
                            FileStream fs = new FileStream(s, FileMode.Open, FileAccess.Read);
                            BinaryFormatter bf = new BinaryFormatter();
                            n = (NeuralNetwork)bf.Deserialize(fs);
                            fs.Close();
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("오류: " + ex);
                    }
                }
                if (selection == "CsvChange")
                {
                    try
                    {
                        FileIO.CSV_Change();
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("오류: " + ex);
                    }
                }
            }

            do
            {
                Console.Write(">");
                selection = Console.ReadLine();
                if (selection == "Help")
                {
                    Console.WriteLine(@"명령어를 알려드립니다.
//신경망 객체가 있습니다.
Help: 이 출력창을 콘솔창에 출력합니다.
ShowStatus: 신경망의 상태를 출력합니다.
CsvTrain: CSV 파일을 통해 신경망을 학습합니다.(첫 문자열에 레이블이 포함되어 있어야합니다.)
CsvQuery: CSV 파일을 통해 신경망에 질의합니다.
ImageTrain: Image파일을 통해 신경망을 학습합니다. 다중선택 가능. (이미지 파일이름의 첫글자를 레이블로 합니다.)
ImageQuery: Image파일을 통해 신경망을 질의합니다. 다중선택 가능.
(이미지 파일이름의 첫글자로 레이블로 하며 숫자가 아니면 -1(오류)을 정답으로 분류합니다.)
Save: 현재 신경망을 저장합니다.
CsvChange: 이미지 파일을 CSV파일로 저장합니다.(이미지 파일이름의 첫글자를 레이블로합니다.)
");
                }
                if (selection == "ShowStatus")
                {
                    n.Show_Status();
                }
                if (selection == "CsvTrain")
                {
                    try
                    {
                        Console.Write("학습 주기: ");
                        Int32 num = Int32.Parse(Console.ReadLine());
                        String csv_training_path;
                        if ((csv_training_path = FileIO.CSV_Open()) != null)
                        {
                            Double[][] csv_training_data = FileIO.CSV_Read(csv_training_path);
                            for (Int32 i = 0; i < num; i++)
                            {
                                foreach (Double[] t_data in csv_training_data)
                                    MI_Train(n, t_data);
                                Console.Write(i + " ");
                            }
                            Console.WriteLine();
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("오류: " + ex);
                    }
                }
                if (selection == "CsvQuery")
                {
                    try
                    {
                        String csv_query_path;
                        if ((csv_query_path = FileIO.CSV_Open()) != null)
                        {
                            Int32 score = 0;
                            Double[][] csv_query_data = FileIO.CSV_Read(csv_query_path);
                            foreach (Double[] q_data in csv_query_data)
                                if (MI_Query(n, q_data)) score++;
                            Console.WriteLine("점수: " + score + "/" + csv_query_data.Length);
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("오류: " + ex);
                    }
                }
                if (selection == "ImageTrain")
                {
                    try
                    {
                        Console.Write("학습 주기: ");
                        Int32 num = Int32.Parse(Console.ReadLine());
                        String[] image_training_list;
                        if ((image_training_list = FileIO.Image_Open()) != null)
                        {
                            for (Int32 i = 0; i < num; i++)
                            {
                                foreach (String record in image_training_list)
                                {
                                    Double[] image_training_data = FileIO.Image_Read(record);
                                    MI_Train(n, image_training_data);
                                }
                                Console.Write(i + " ");
                            }
                            Console.WriteLine();
                        }

                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("오류: " + ex);
                    }

                }

                if (selection == "ImageQuery")
                {
                    try
                    {
                        String[] image_query_list;
                        if ((image_query_list = FileIO.Image_Open()) != null)
                        {
                            Int32 score = 0;
                            foreach (String record in image_query_list)
                            {
                                Double[] image_query_data = FileIO.Image_Read(record);
                                if (MI_Query(n, image_query_data)) score++;
                            }
                            Console.WriteLine("점수: " + score + "/" + image_query_list.Length);
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("오류: " + ex);
                    }
                }
                if (selection == "Save")
                {
                    try
                    {
                        String s;
                        if ((s = FileIO.Neural_Save()) != null)
                        {
                            FileStream fs = new FileStream(s, FileMode.Create, FileAccess.Write);
                            BinaryFormatter bf = new BinaryFormatter();
                            bf.Serialize(fs, n);
                            fs.Close();
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("오류: " + ex);
                    }
                }
                if (selection == "CsvChange")
                {
                    try
                    {
                        FileIO.CSV_Change();
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("오류: " + ex);
                    }
                }
            } while (selection != "q");
        }
    }
}