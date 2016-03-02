#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "cJoystick.h"
#include <iostream>

#define JOYSTICK_DEV "/dev/input/js0"

cJoystick::cJoystick() {
    active = false;
    joystick_fd = 0;
    joystick_ev = new js_event();
    joystick_st = new joystick_state();
    joystick_fd = open(JOYSTICK_DEV, O_RDONLY | O_NONBLOCK);
    if (joystick_fd > 0) {
        ioctl(joystick_fd, JSIOCGNAME(256), name);
        ioctl(joystick_fd, JSIOCGVERSION, &version);
        ioctl(joystick_fd, JSIOCGAXES, &axes);
        ioctl(joystick_fd, JSIOCGBUTTONS, &buttons);
        std::cout << "   Name: " << name << std::endl;
        std::cout << "Version: " << version << std::endl;
        std::cout << "   Axes: " << (int)axes << std::endl;
        std::cout << "Buttons: " << (int)buttons << std::endl;
        joystick_st->axis.reserve(axes);
        joystick_st->button.reserve(buttons);
        active = true;
    }
}


int cJoystick::readEv() {
    int bytes = read(joystick_fd, joystick_ev, sizeof(*joystick_ev));
    if (bytes > 0) {
        joystick_ev->type &= ~JS_EVENT_INIT;
        if (joystick_ev->type & JS_EVENT_BUTTON) {
            joystick_st->button[joystick_ev->number] = joystick_ev->value;
            //std::cout << (int)joystick_ev->number << std::endl;
        }
        if (joystick_ev->type & JS_EVENT_AXIS) {
            joystick_st->axis[joystick_ev->number] = joystick_ev->value;
        }
    }
    return bytes;
}

cJoystick::~cJoystick() {
    if (joystick_fd > 0) {
        active = false;
        close(joystick_fd);
    }
    delete joystick_st;
    delete joystick_ev;
    joystick_fd = 0;
}
