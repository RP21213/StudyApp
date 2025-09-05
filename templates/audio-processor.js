/**
 * This class is an AudioWorkletProcessor.
 * It runs in a separate, high-priority thread to receive raw audio data
 * from the microphone and efficiently pass it to the main application thread.
 * This avoids blocking the UI and prevents audio dropouts.
 */
class AudioRecorderProcessor extends AudioWorkletProcessor {
  /**
   * The process method is called by the browser's audio engine whenever a new
   * block of audio data is available.
   * @param {Float32Array[][]} inputs - An array of inputs, each with an array of channels.
   *                                    We typically only use inputs[0][0] for mono audio.
   * @returns {boolean} - Must return true to keep the processor alive.
   */
  process(inputs) {
    // We expect a single input with a single channel (mono).
    const inputChannelData = inputs[0][0];

    // If the channel data is empty for any reason, stop processing.
    if (!inputChannelData) {
      return true;
    }

    // Post the raw Float32Array audio data back to the main thread.
    // We send a clone of the data to avoid issues with memory transfer.
    this.port.postMessage(inputChannelData.slice(0));

    // Return true to indicate that the processor should continue running.
    return true;
  }
}

// Register the processor with the browser, giving it a name that can be
// referenced from the main thread when creating an AudioWorkletNode.
registerProcessor('audio-recorder-processor', AudioRecorderProcessor);