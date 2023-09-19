#include "simple_apfc.hpp"


const std::string SimpleAPFC::name = "apfc";


SimpleAPFC::SimpleAPFC() {
    this->probSpace = ProblemStat(this->name + "->space");
    ProblemInstat(this->name, probSpace);
}


void SimpleAPFC::setTime(AdaptInfo* adaptInfo) {
    ProblemInstat::setTime(adaptInfo);

    this->measure();
    this->writeOutput();
}


void SimpleAPFC::setupOperators() {

}
