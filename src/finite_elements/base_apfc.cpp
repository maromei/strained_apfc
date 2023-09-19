#include "base_apfc.hpp"

const std::string BaseAPFC::name = "apfc";


BaseAPFC::BaseAPFC(ProblemStat& probSpace) :
    ProblemInstat("apfc", probSpace),
    m_probSpace(probSpace)
{}


void BaseAPFC::setTime(AdaptInfo* adaptInfo) {
    ProblemInstat::setTime(adaptInfo);

    this->measure();
    this->writeOutput();
}
