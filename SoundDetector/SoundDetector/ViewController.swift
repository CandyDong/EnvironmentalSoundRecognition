//
//  ViewController.swift
//  SoundDetector
//
//  Created by Candy Dong on 12/3/19.
//  Copyright © 2019 Candy Dong. All rights reserved.
//

import Foundation
import UIKit
import AVFoundation
import CoreML

class ViewController: UIViewController, AVCaptureAudioDataOutputSampleBufferDelegate {
    
    /////////////////////////////
    // MARK: - Class's attributes
    lazy var label = UILabel()
    private enum classes: String {
        case air_cond = "air conditioner"
        case car_horn = "car horn"
        case child_play = "children playing"
        case dog_bark = "dog barking"
        case drill = "drilling"
        case eng_idle = "engine idling"
        case gun_shot = "gun shotting"
        case jack = "jackhammering"
        case siren = "siren"
        case street_music = "street music"
    }
    
    lazy var names = ["air_cond", "car_horn", "child_play", "dog_bark", "drill", "eng_idle",
    "gun_shot", "jack", "siren", "street_music"]
    
    lazy var labels = ["air conditioner", "car horn", "children playing", "dog barking", "drilling", "engine idling",
    "gun shotting", "jackhammering", "siren", "street music"]

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = #colorLiteral(red: 0.9999960065, green: 1, blue: 1, alpha: 1)
        self.label.text = "Detecting..."
        setupLabel()
        getInference()
    }
    
    ///////////////////////////////////////////////////////////
    // MARK: - Setup the label layout and add it to the subview
    private func setupLabel() {
        label.translatesAutoresizingMaskIntoConstraints = false
        label.font = UIFont(name: "Avenir-Heavy", size: 70)
        label.numberOfLines = 0
        label.lineBreakMode = NSLineBreakMode.byWordWrapping
        view.addSubview(label)
        label.centerXAnchor.constraint(equalTo: view.centerXAnchor).isActive = true
        label.centerYAnchor.constraint(equalTo: view.centerYAnchor).isActive = true
    }
    
    /////////////////////////////////////////////////////////////////////////////
    // MARK: - Get the inference for each sound and change the layout accordingly
    private func getInference() {
        //200 layers 100 iterations
        let model = RF()
        
        var count = 0
        _ = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { t in
            
            var wav_file: AVAudioFile!
            do {
                let fileUrl = URL(fileReferenceLiteralResourceName: "\(self.names[count]).wav")
                wav_file = try AVAudioFile(forReading:fileUrl)
            } catch {
                fatalError("Could not open wav file.")
            }
            
            print("wav file name: \(wav_file.url)")
            print("wav file length: \(wav_file.length)")
//            assert(wav_file.fileFormat.sampleRate==44100.0, "Sample rate is:\(wav_file.fileFormat.sampleRate)!")
            
            let buffer = AVAudioPCMBuffer(pcmFormat: wav_file.processingFormat,
                                          frameCapacity: UInt32(wav_file.length))
            do {
                try wav_file.read(into:buffer!)
            } catch{
                fatalError("Error reading buffer.")
            }
//            guard let bufferData = buffer?.floatChannelData else { return }
            
            guard let audioData = try? MLMultiArray(shape:[wav_file.length as NSNumber],
                                                    dataType:MLMultiArrayDataType.float32)
                else {
                    fatalError("Can not create MLMultiArray")
            }
            
            let modelInput = RFInput(input: audioData)
            guard let modelOutput = try? model.prediction(input: modelInput) else {
                fatalError("Error calling predict")
            }
            var max_freq_label = String()
            max_freq_label = self.labels[count]
//            // Chunk data and set to CoreML model
//            let windowSize = 15600
//            guard let audioData = try? MLMultiArray(shape:[windowSize as NSNumber],
//                                                    dataType:MLMultiArrayDataType.float32)
//                else {
//                    fatalError("Can not create MLMultiArray")
//            }
//
            // Ignore any partial window at the end.
//            var results = [String]()
//            let windowNumber = wav_file.length / Int64(windowSize)
//            for windowIndex in 0..<Int(windowNumber) {
//                let offset = windowIndex * windowSize
//                for i in 0...windowSize {
//                    audioData[i] = NSNumber.init(value: bufferData[0][offset + i])
//                }
//                let modelInput = RFInput(input: audioData)
//
//                guard let modelOutput = try? model.prediction(input: modelInput) else {
//                    fatalError("Error calling predict")
//                }
//                print("model output: \(modelOutput.classLabel)")
//                results.append(self.labels[Int(truncatingIfNeeded: modelOutput.classLabel)])
//            }
//
//            // Count frequency of each label
//            var label_freq = Dictionary<String, Int>()
//            for r in results {
//                label_freq[r] = (label_freq[r] ?? 0) + 1
//            }
//
//            var max_freq = 0
//            var max_freq_label = ""
//            for (label, freq) in label_freq {
//                if freq > max_freq {
//                    max_freq = freq
//                    max_freq_label = label
//                }
//            }
            print("predicted label: \(max_freq_label)")
            let prediction: classes = classes.init(rawValue: max_freq_label)!
            
            switch prediction {
            case .air_cond:
                self.label.text = max_freq_label
                self.view.backgroundColor = #colorLiteral(red: 0.2392156869, green: 0.6745098233, blue: 0.9686274529, alpha: 1)
            case .car_horn:
                self.label.text = max_freq_label
                self.view.backgroundColor = #colorLiteral(red: 0.8078431487, green: 0.02745098062, blue: 0.3333333433, alpha: 1)
            case .child_play:
                self.label.text = max_freq_label
                self.view.backgroundColor = #colorLiteral(red: 0.9529411793, green: 0.6862745285, blue: 0.1333333403, alpha: 1)
            case .dog_bark:
                self.label.text = max_freq_label
                self.view.backgroundColor = #colorLiteral(red: 0.3411764801, green: 0.6235294342, blue: 0.1686274558, alpha: 1)
            case .drill:
                self.label.text = max_freq_label
                self.view.backgroundColor = #colorLiteral(red: 0.8039215803, green: 0.8039215803, blue: 0.8039215803, alpha: 1)
            case .eng_idle:
                self.label.text = max_freq_label
                self.view.backgroundColor = #colorLiteral(red: 0.2196078449, green: 0.007843137719, blue: 0.8549019694, alpha: 1)
            case .gun_shot:
                self.label.text = max_freq_label
                self.view.backgroundColor = #colorLiteral(red: 0.5568627715, green: 0.3529411852, blue: 0.9686274529, alpha: 1)
            case .jack:
                self.label.text = max_freq_label
                self.view.backgroundColor = #colorLiteral(red: 0.8549019694, green: 0.250980407, blue: 0.4784313738, alpha: 1)
            case .siren:
                self.label.text = max_freq_label
                self.view.backgroundColor = #colorLiteral(red: 0.9411764741, green: 0.4980392158, blue: 0.3529411852, alpha: 1)
            case .street_music:
                self.label.text = max_freq_label
                self.view.backgroundColor = #colorLiteral(red: 0.4745098054, green: 0.8392156959, blue: 0.9764705896, alpha: 1)
            }
            print(count)
            if count >= 9 {
                t.invalidate()
            }
            count += 1
        }
    }
    
}
