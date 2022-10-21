class RecorderProcessor extends AudioWorkletProcessor {

  constructor() {
    super()
    this.buffer_size = 16000 * 2 // 16 kHz with 2 second time resolution for short breaks
    this.buffer = new Float32Array(this.buffer_size)
    this.init_buffer()
    this.volume_frames = 0
    this.square_sum = 0
  }

  init_buffer() {
    this.samples_written = 0
  }

  is_buffer_empty() {
    return this.samples_written === 0
  }

  is_buffer_full() {
    return this.samples_written === this.buffer_size
  }

  append(channels) {
    const samples = channels[0].length
    for (let i = 0; i < samples; i++) {
      if (i % 3 !== 0) {
        // simple conversion from 48 kHz to 16 kHz by skipping 2 samples,
        // reduce sampling rate here in worker to reduce data transmissions
        continue
      }
      let channel_sample_sum = 0
      for (let j = 0; j < channels.length; ++j) {
        const sample = channels[j][i]
        channel_sample_sum += sample
      }
      const sample = channel_sample_sum / channels.length
      const previous_sample = this.buffer[this.samples_written]
      this.square_sum += sample * sample - previous_sample * previous_sample
      this.buffer[this.samples_written++] = sample
      if (this.is_buffer_full()) {
        this.flush()
      }
    }
  }

  flush() {
    const rms = Math.sqrt(this.square_sum / this.samples_written)
    // ignore silent blocks
    if (rms > 0.01) {
      this.port.postMessage(this.buffer.slice(0, this.samples_written))
    }
    this.init_buffer()
  }

  process(inputs) {
    const channels = inputs[0]
    if (!channels[0]) {
      return
    }
    // input[n'th input][m'th channel] -> Float32Array with 128 sample values in range [-1 .. 1]
    // 128 samples at 48 kHz is is 2.66 ms
    this.append(channels)
    // calculate and send volume information
    this.volume_frames += 128
    if (this.volume_frames > 12800) {
      const rms = Math.sqrt(this.square_sum / this.samples_written)
      const db = 20 * Math.log10(rms * 100)
      this.port.postMessage(
        db
      )
      this.volume_frames = 0
    }
    return true // keep processor alive
  }

}

registerProcessor("recorder.worklet", RecorderProcessor)
