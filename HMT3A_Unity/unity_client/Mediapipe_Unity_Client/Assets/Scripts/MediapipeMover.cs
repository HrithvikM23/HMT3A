using UnityEngine;

public class MediapipeMover : MonoBehaviour
{
    public UDPReceiver receiver;

    public Transform hips;
    public Transform leftShoulder;
    public Transform rightShoulder;

    void Start()
    {
        if (receiver == null)
            receiver = FindObjectOfType<UDPReceiver>();
    }

    void Update()
    {
        if (receiver == null || receiver.latestLandmarks == null)
            return;

        // Example mapping (very first test)
        if (receiver.latestLandmarks.ContainsKey(0)) // nose
        {
            Landmark nose = receiver.latestLandmarks[0];

            // move whole character slightly based on your position
            transform.position = new Vector3(
                (nose.x - 0.5f) * 2f,
                0,
                (nose.y - 0.5f) * 2f
            );
        }
    }
}
