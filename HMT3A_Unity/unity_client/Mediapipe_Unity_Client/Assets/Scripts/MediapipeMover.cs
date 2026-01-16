using UnityEngine;

public class MediapipeMover : MonoBehaviour
{
    public UDPReceiver receiver;

    void Start()
    {
        if (receiver == null)
            receiver = FindObjectOfType<UDPReceiver>();

        if (receiver == null)
            Debug.LogError("No UDPReceiver found in scene.");
    }

    void Update()
    {
        if (receiver == null || receiver.latestLandmarks == null)
            return;

        // Simple first test: move character using the NOSE (landmark 0)
        if (receiver.latestLandmarks.ContainsKey(0))
        {
            Landmark nose = receiver.latestLandmarks[0];

            transform.position = new Vector3(
                (nose.x - 0.5f) * 3f,
                transform.position.y,
                (nose.y - 0.5f) * 3f
            );
        }
    }
}
