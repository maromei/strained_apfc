#pragma once

#include <string>

#include "AMDiS.h"
#include "file_writer.hpp"
#include "measurements.hpp"
#include "initialiser.hpp"

using namespace AMDiS;

class SimpleAPFC : public ProblemInstat {

    public:

    static const std::string name;

    SimpleAPFC();

    void writeOutput();
    void measure();

    /**
     * @brief Set the Time
     *
     * Overridden from the `ProblemInstat` just to run the
     * `measure()` and `writeOutput()` on each timestep.
     *
     * @param adaptInfo
     */
    void setTime(AdaptInfo* adaptInfo);

    void setupOperators();
    void setupSpace();

    private:

    ProblemStat probSpace;


};
