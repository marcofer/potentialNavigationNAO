#include <vector>
#include "/usr/include/linux/joystick.h"

struct joystick_state {
    std::vector<signed short> button;
    std::vector<signed short> axis;
};

class cJoystick {
    private:
        pthread_t thread;
        bool active;
        int joystick_fd;
        js_event *joystick_ev;
        joystick_state *joystick_st;
        __u32 version;
        __u8 axes;
        __u8 buttons;
        char name[256];
    protected:
    public:
        cJoystick();
        ~cJoystick();
        static void* loop(void* obj);
        int readEv();

        inline js_event* getJsEventPtr() {return joystick_ev; }
        inline joystick_state* getJsStatePtr() {return joystick_st;}
};

