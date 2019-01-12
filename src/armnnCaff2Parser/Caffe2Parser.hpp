#pragma once
#include "armnnCaffe2Parser/ICaffe2Parser.hpp"

#include "armnn/Types.hpp"
#include "armnn/NetworkFwd.hpp"
#include "armnn/Tensor.hpp"

#include <memory>
#include <vector>
#include <unordered_map>
#include "caffe2.pb.h"





namespace armnnCaffe2Parser
{

    using BindingPointInfo = std::pair<armnn::LayerBindingId, armnn::TensorInfo>;
    
    class Caffe2ParserBase: public ICaffe2Parser
    {
     public:
        
         void CreateNetworkFromNetDef(caffe2::NetDef& init,caffe2::NetDef& predict,const std::map<std::string, armnn::TensorShape>& inputShapes);
         
        Caffe2ParserBase();
        
    protected:

        void ParseInputLayer();
        void ParseReluLayer(const caffe2::OperatorDef& op);
<<<<<<< HEAD
        void ParseSoftmaxLayer(const caffe2::OperatorDef& op);
         void ParsePoolingLayer(const caffe2::OperatorDef& op);
=======
        void ParseFCLayer(const caffe2::OperatorDef& op);
>>>>>>> 2207b1c4a601cb5b03662dc5b47ad390117a66ed



    
    armnn::TensorInfo  ArgumentToTensorInfo(const caffe2::Argument& arg);
    armnn::IOutputSlot& GetArmnnOutputSlotForCaffe2Output(const std::string& caffe2outputName) const;  
    void SetArmnnOutputSlotForCaffe2Output(const std::string& caffe2OutputName, armnn::IOutputSlot& armnnOutputSlot);
    void TrackBindingPoint(armnn::IConnectableLayer* layer,
    armnn::LayerBindingId id,
    const armnn::TensorInfo& tensorInfo,
    const char* bindingPointDesc,
    std::unordered_map<std::string, BindingPointInfo>& nameToBindingInfo);

        void TrackInputBinding(armnn::IConnectableLayer* layer,
            armnn::LayerBindingId id,
            const armnn::TensorInfo& tensorInfo);
        void LoadNetDef(caffe2::NetDef& init,caffe2::NetDef& predict);
        armnn::INetworkPtr m_Network;

        std::map<std::string, const caffe2::OperatorDef*> m_Caffe2OperatorsByOutputName;
        using OperationParsingFunction = void(Caffe2ParserBase::*)(const caffe2::OperatorDef op);
        static const std::map<std::string, OperationParsingFunction> ms_Caffe2OperatorToParsingFunctions;
        std::map<std::string, armnn::TensorShape> m_InputShapes;

        std::unordered_map<std::string, BindingPointInfo> m_NetworkInputsBindingInfo;

        /// As we add armnn layers we store the armnn IOutputSlot which corresponds to the Caffe2 tops.
        std::unordered_map<std::string, armnn::IOutputSlot*> m_ArmnnOutputSlotForCaffe2Output;

        std::map<std::string, const caffe2::OperatorDef*> blobs;

/*
        /// Retrieves binding info (layer id and tensor info) for the network input identified by the given layer name.
    virtual BindingPointInfo GetNetworkInputBindingInfo(const std::string& name) const override;

    /// Retrieves binding info (layer id and tensor info) for the network output identified by the given layer name.
    virtual BindingPointInfo GetNetworkOutputBindingInfo(const std::string& name) const override;
    Caffe2ParserBase();*/
    };


    class Caffe2Parser : public Caffe2ParserBase
    {
    public:

        virtual void CreateNetworkFromBinaryFile(const char* predict_net,const char* init_net,
                const std::map<std::string, armnn::TensorShape>& inputShapes)override;
    // public:
         Caffe2Parser();
    };
    }

 