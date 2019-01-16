#pragma once

#include "armnn/Types.hpp"
#include "armnn/NetworkFwd.hpp"
#include "armnn/Tensor.hpp"
#include "armnn/INetwork.hpp"

#include <memory>
#include <map>
#include <vector>


namespace armnnCaffe2Parser
{



class ICaffe2Parser;


class ICaffe2Parser
{
public:
    static ICaffe2Parser* Create();
    static void Destroy(ICaffe2Parser* parser);


    /// Create the network from a protobuf binary file on the disk.
    virtual void CreateNetworkFromBinaryFile(const char* predict_net,const char* init_net,
                const std::map<std::string, armnn::TensorShape>& inputShapes) = 0;

   
/*
    /// Retrieve binding info (layer id and tensor info) for the network input identified by the given layer name.
    virtual BindingPointInfo GetNetworkInputBindingInfo(const std::string& name) const = 0;

    /// Retrieve binding info (layer id and tensor info) for the network output identified by the given layer name.
    virtual BindingPointInfo GetNetworkOutputBindingInfo(const std::string& name) const = 0;
*/
protected:
    virtual ~ICaffe2Parser() {};
};

}