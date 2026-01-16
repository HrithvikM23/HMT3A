using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

[Serializable]
public class Landmark
{
    public float x;
    public float y;
    public float z;
}

public class UDPReceiver : MonoBehaviour
{
    private UdpClient client;
    private const int PORT = 5052;

    // THIS is what MediapipeMover will read
    public Dictionary<int, Landmark> latestLandmarks =
        new Dictionary<int, Landmark>();

    void Start()
    {
        client = new UdpClient(PORT);
        client.BeginReceive(ReceiveCallback, null);
        Debug.Log("Listening on UDP port 5052");
    }

    void ReceiveCallback(IAsyncResult result)
    {
        IPEndPoint ip = new IPEndPoint(IPAddress.Any, PORT);
        byte[] data = client.EndReceive(result, ref ip);

        string json = Encoding.UTF8.GetString(data);

        try
        {
            latestLandmarks = JsonUtility.FromJson<Wrapper>(Wrap(json)).dict;
        }
        catch (Exception e)
        {
            Debug.LogWarning("JSON parse failed: " + e.Message);
        }

        client.BeginReceive(ReceiveCallback, null);
    }

    // Unity cannot parse raw dictionaries, so we wrap it
    string Wrap(string json)
    {
        return "{\"dict\":" + json + "}";
    }

    [Serializable]
    public class Wrapper
    {
        public Dictionary<int, Landmark> dict;
    }

    void OnApplicationQuit()
    {
        client.Close();
    }
}
