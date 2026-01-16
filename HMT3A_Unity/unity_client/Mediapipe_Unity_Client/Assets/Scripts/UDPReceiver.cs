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
    UdpClient client;
    int port = 5052;

    public Dictionary<int, Landmark> latestLandmarks =
        new Dictionary<int, Landmark>();

    void Start()
    {
        client = new UdpClient(port);
        client.BeginReceive(ReceiveCallback, null);
        Debug.Log("UDP Receiver started on port " + port);
    }

    void ReceiveCallback(IAsyncResult result)
    {
        IPEndPoint ip = new IPEndPoint(IPAddress.Any, port);
        byte[] data = client.EndReceive(result, ref ip);

        string json = Encoding.UTF8.GetString(data);

        try
        {
            latestLandmarks = JsonUtility.FromJson<Wrapper>(Wrap(json)).dict;
        }
        catch { }

        client.BeginReceive(ReceiveCallback, null);
    }

    // Helper to make Unity's JSON work with dictionary
    string Wrap(string json)
    {
        return "{\"dict\":" + json + "}";
    }

    void OnApplicationQuit()
    {
        client.Close();
    }

    [Serializable]
    public class Wrapper
    {
        public Dictionary<int, Landmark> dict;
    }
}
