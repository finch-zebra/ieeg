#include <SDL2/SDL.h>
#include <SDL_audio.h>
#include <wiringPi.h>

#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "AudioFile.h"

#define ASSERT(cond, msg)                                                                                             \
    if (!static_cast<bool>(cond))                                                                                     \
    {                                                                                                                 \
        std::ostringstream stream;                                                                                    \
        stream << "Failure at " << __FILE__ << ":" << __LINE__ << " in \"" << #cond << "\":\n\t" << msg << std::endl; \
        throw std::runtime_error(stream.str());                                                                       \
    }

#define POWER 0.75
#define TRIALS 15
#define OUTPUT_PIN 21
#define DELAY_MIN 4000
#define DELAY_MAX 6000
#define SAMPLE_RATE 44100

using namespace std::chrono;

template <typename T>
using Vector = std::vector<T>;

template <typename K, typename V>
using Map = std::map<K, V>;

using Size = size_t;
using Sample = float;
using String = std::string;
using Samples = Vector<Sample>;

template <typename T>
void print(T &&value)
{
    std::cout << value << std::endl;
}
template <typename Head, typename... Tail>
void print(Head &&head, Tail &&...tail)
{
    std::cout << head << " ";
    print(std::forward<Tail>(tail)...);
}
void print() { std::cout << std::endl; }

struct Timer
{
    high_resolution_clock::time_point time;

    Timer() { reset(); }

    double operator()()
    {
        auto now = high_resolution_clock::now();
        auto elapsed = duration_cast<milliseconds>(now - time);
        time = now;
        return double(elapsed.count());
    }

    void reset() { time = high_resolution_clock::now(); }
};

struct Pins
{
    Pins()
    {
        wiringPiSetup();
        pinMode(OUTPUT_PIN, OUTPUT);
    }

    void set(bool high)  //
    {
        digitalWrite(OUTPUT_PIN, high ? HIGH : LOW);
    }
};

struct Speaker
{
    SDL_AudioDeviceID device;

    Speaker()
    {
        SDL_Init(SDL_INIT_AUDIO);

        SDL_AudioSpec wantspec, havespec;
        wantspec.freq = SAMPLE_RATE;
        wantspec.format = sample_format();
        wantspec.channels = 1;
        wantspec.samples = 0;
        wantspec.callback = nullptr;
        wantspec.userdata = nullptr;

        device = SDL_OpenAudioDevice(nullptr, 0, &wantspec, &havespec, 0);
        ASSERT(device != 0, "Failed to open SDL audio device: " << SDL_GetError());
    }

    SDL_AudioFormat sample_format()
    {
        if (std::is_same<Sample, float>()) return AUDIO_F32;
        if (std::is_same<Sample, uint8_t>()) return AUDIO_U8;
        if (std::is_same<Sample, int8_t>()) return AUDIO_S8;
        if (std::is_same<Sample, uint16_t>()) return AUDIO_U16;
        if (std::is_same<Sample, int16_t>()) return AUDIO_S16;
        if (std::is_same<Sample, int32_t>()) return AUDIO_S32;
        ASSERT(false, "Failed to get format of sample type!");
    }

    ~Speaker() { SDL_CloseAudioDevice(device); }
    void queue(const Samples &data) { SDL_QueueAudio(device, data.data(), data.size() * sizeof(Sample)); }
    void play() { SDL_PauseAudioDevice(device, 0); }
    void pause() { SDL_PauseAudioDevice(device, 1); }
    int playing() { return SDL_GetAudioDeviceStatus(device); }
};

enum class Type
{
    ColoredNoise,
    BirdsOwnSong,
    Conspecific,
    ReverseBos,
    Tone5k,
};

String type_name(Type type)
{
    if (type == Type::ColoredNoise) return "Colored Noise";
    if (type == Type::BirdsOwnSong) return "Bird's Own Song";
    if (type == Type::Conspecific) return "Conspecific Song";
    if (type == Type::ReverseBos) return "Reversed BOS";
    if (type == Type::Tone5k) return "5K Tone";
    ASSERT(false, "Invalid type given");
}

String type_path(Type type)
{
    if (type == Type::ColoredNoise) return "color_noise";
    if (type == Type::BirdsOwnSong) return "song_bos";
    if (type == Type::Conspecific) return "song_con";
    if (type == Type::ReverseBos) return "song_rev";
    if (type == Type::Tone5k) return "tone_5k";
    ASSERT(false, "Invalid type given");
}

struct Event
{
    Type type;
    int desc = -1;
    double msec = 0;
    Size index = 0;
    bool start = false;

    void action(Pins &pins) const
    {
        pins.set(start);
        String start = this->start ? "Starting" : "Ending";
        print(start + " " + type_name(type));
    }
};

struct Control
{
    Pins &pins;
    Vector<Event> list;

    Control(Pins &pins) : pins(pins) {}

    void add(const Event &info) { list.push_back(info); }

    void run()
    {
        double previous = 0;
        auto start = high_resolution_clock::now();

        for (const auto &info : list)
        {
            auto now = high_resolution_clock::now();
            auto time = duration_cast<milliseconds>(now - start).count();

            Size interval = info.msec - time;
            std::this_thread::sleep_for(milliseconds(interval));
            now = high_resolution_clock::now();

            time = duration_cast<milliseconds>(now - start).count();
            print("Duration:", info.msec - previous, ", Difference:", info.msec - time);
            previous = info.msec;

            info.action(pins);
        }
    }
};

Samples generate_silence(Size duration)
{
    Size frames = SAMPLE_RATE * duration / 1000;
    return Samples(frames, 0);
}

Samples &append(Samples &samples, const Samples &other)
{
    samples.insert(samples.end(), other.begin(), other.end());
    return samples;
}

Samples generate_task(Control &timers, Map<Type, Samples> &songs)
{
    std::vector<Type> order;
    for (const auto &pair : songs)
        for (Size _ = 0; _ < TRIALS; ++_) order.push_back(pair.first);
    std::shuffle(order.begin(), order.end(), std::default_random_engine(rand()));

    Samples samples;
    for (Size i = 0; i < order.size(); ++i)
    {
        Event event1{.type = order[i], .index = i, .start = true};
        event1.msec = samples.size() * 1000.0 / SAMPLE_RATE;
        timers.add(event1);

        Event event2{.type = order[i], .index = i, .start = false};
        event2.msec = event1.msec + (Size(event2.type) + 1) * 100;
        timers.add(event2);

        append(samples, songs[order[i]]);

        if (i != order.size() - 1)
        {
            Size range = (DELAY_MAX - DELAY_MIN + 1);
            auto silence = generate_silence(DELAY_MIN + (rand() % range));
            append(samples, silence);
        }
    }
    return samples;
}

Samples load_audio(const String &path)
{
    AudioFile<Sample> audio;
    ASSERT(audio.load(path), "Failed to load audio file " << path << "!");
    ASSERT(audio.getNumChannels() == 1, "Audio is not mono channel!");
    ASSERT(audio.getSampleRate() == SAMPLE_RATE, "Audio sample rate is incorrect!");
    Samples samples = audio.samples[0];
    for (auto &sample : samples) sample *= POWER;
    return samples;
}

int main(int argc, char *argv[])
{
    SDL_Init(SDL_INIT_AUDIO);
    srand(time(nullptr));

    ASSERT(argc == 2, "The songs folder must be passed to the program!");

    Pins pins;
    Speaker speaker;
    Control control(pins);

    Map<Type, Samples> songs;
    auto path = argv[1] + String("/");
    for (auto type : {Type::ColoredNoise, Type::BirdsOwnSong,  //
                      Type::Conspecific, Type::ReverseBos, Type::Tone5k})
        songs[type] = load_audio(path + type_path(type) + ".wav");

    Timer T;
    auto samples = generate_task(control, songs);
    speaker.queue(samples);
    auto elapsed = T();

    print("Startup:", elapsed, "Milliseconds");
    print("Length:", samples.size() / double(SAMPLE_RATE), "Seconds");
    print("Size:", samples.size() * 4.0 / 1024 / 1024, "Megabytes");
    print();

    speaker.play();
    control.run();
    print("Done!");
    std::this_thread::sleep_for(milliseconds(1000));
    return 0;
}
