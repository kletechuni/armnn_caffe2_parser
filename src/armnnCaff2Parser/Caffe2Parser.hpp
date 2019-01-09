#pragma once
#include "armnnCaffe2Parser/ICaffe2Parser.hpp"

#include "armnn/Types.hpp"
#include "armnn/NetworkFwd.hpp"
#include "armnn/Tensor.hpp"

#include <memory>
#include <vector>
#include <unordered_map>



namespace caffe2
{
class NetDef;
};


namespace armnnCaffe2Parser
{
    
    class Caffe2ParserBase: public ICaffe2Parser
    {
     public:
         void CreateNetworkFromBinaryFile(const char* graphFile)override;
         void CreateNetworkFromNetDef(caffe2::NetDef& netDef);
         
        Caffe2ParserBase()
        {}
    protected:
        void LoadNetDef(caffe2::NetDef& netDef);
        armnn::INetworkPtr m_Network;

        std::map<std::string, const caffe2::OperatorDef*> m_Caffe2OperatorsByOutputName;
        using OperationParsingFunction = void(Caffe2ParserBase::*)(const caffe2::OperatorDef op);
        static const std::map<std::string, OperationParsingFunction> ms_Caffe2OperatorToParsingFunctions;

/*
        /// Retrieves binding info (layer id and tensor info) for the network input identified by the given layer name.
    virtual BindingPointInfo GetNetworkInputBindingInfo(const std::string& name) const override;

    /// Retrieves binding info (layer id and tensor info) for the network output identified by the given layer name.
    virtual BindingPointInfo GetNetworkOutputBindingInfo(const std::string& name) const override;
    Caffe2ParserBase();*/
    };
    }

 