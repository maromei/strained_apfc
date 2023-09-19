#pragma once

#include <string>

#include "AMDiS.h"
#include "file_writer.hpp"
#include "measurements.hpp"
#include "initialiser.hpp"

using namespace AMDiS;

class BaseAPFC : public ProblemInstat {

    public:

    static const std::string name;

    BaseAPFC(ProblemStat& probSpace);

    virtual void writeOutput() {};
    virtual void measure() {};

    /**
     * @brief Set the Time
     *
     * Overridden from the `ProblemInstat` just to run the
     * `measure()` and `writeOutput()` on each timestep.
     *
     * @param adaptInfo
     */
    void setTime(AdaptInfo* adaptInfo);

    virtual void setupOperators() {};
    virtual void setupSpace() {};

    private:

    ProblemStat m_probSpace;
};
