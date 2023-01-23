#ifndef __AMS_DATA_BROKER_HPP__
#define __AMS_DATA_BROKER_HPP__

namespace ams {

/*
 * Abstract class that defines what AMS expect from a message broker.
 */

class DataBroker {
public:
      /* Define the type of the broker (RabbitMQ etc) */
      virtual std::string type() const = 0;
      virtual ~DataBroker() {}
};

} //namespace ams

#endif
