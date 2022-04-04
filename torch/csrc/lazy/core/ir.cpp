#include <torch/csrc/lazy/core/ir.h>

#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include "lazy/core/hash.h"

C10_DEFINE_bool(ltc_enable_dynamic_shapes, false, "Whether dynamic shape is enabled");

namespace torch {
namespace lazy {

std::vector<NodePtr> Node::last_node_list;
std::vector<NodePtr> Node::node_list;

size_t Output::Hasher::operator()(const Output& output) const {
  return StdHashCombine(
      reinterpret_cast<std::ptrdiff_t>(output.node), output.index);
}

bool Output::operator==(const Value& rhs) const {
  printf(
      "Calling Output::operator==,%s, op=%s, comparing to index %lu and %lu \n",
      HashToString(node->hash()).c_str(),
      HashToString(rhs.node()->hash()).c_str(),
      index,
      rhs.index);
  return false;
  return node->hash() == rhs.node()->hash() && index == rhs.index;
}

hash_t Output::hash() const {
  return HashCombine(node->hash(), Hash(index));
}

std::string Output::ToString() const {
  std::stringstream ss;
  ss << node->ToString() << ", index=" << index;
  return ss.str();
}

hash_t Value::hash() const {
  return HashCombine(node()->hash(), Hash(index));
}

hash_t Value::hash_with_sizes() const {
  return HashCombine(node()->hash_with_sizes(), Hash(index));
}

hash_t Value::hash_without_sizes() const {
  return HashCombine(node()->hash_without_sizes(), Hash(index));
}

OpKind OpKind::Get(const std::string& name) {
  return OpKind(c10::Symbol::fromQualString(name));
}

hash_t OpKind::hash() const {
  return StringHash(op.toQualString());
}

bool Node::enableDynamicShape() {
  static bool enabled = std::getenv("LTC_ENABLE_DYNAMIC_SHAPES") != nullptr;
  return enabled || FLAGS_ltc_enable_dynamic_shapes;
}

size_t Node::NextNodeListIndex() {
  // Tracing is done in a single thread, so no need to use atomic here.
  return node_list.size();
}

void Node::PushIntoNodeList(NodePtr node) {
  // printf("Pushing %s into index %lu\n", node.get()->op().ToString().c_str(),
  // NextNodeListIndex());
  node_list.push_back(node);
}

void Node::ClearNodeList() {
  // printf("Node::ClearNodeList\n");
  last_node_list.clear();
  std::swap(last_node_list, node_list);
}

Node::Node(
    OpKind op,
    size_t num_outputs,
    hash_t node_hash,
    std::function<hash_t(bool)> dag_hash_fn)
    : op_(op),
      num_outputs_(num_outputs),
      node_hash_(node_hash),
      dag_hash_without_sizes_(dag_hash_fn(false)),
      dag_hash_with_sizes_(dag_hash_fn(true)),
      metadata_(GetMetaDataIfDebugging()),
      node_list_index_(NextNodeListIndex()) {}

Node::Node(
    OpKind op,
    size_t num_outputs,
    std::function<hash_t(bool)> node_hash_fn)
    : op_(op),
      num_outputs_(num_outputs),
      node_hash_(node_hash_fn(!enableDynamicShape())),
      dag_hash_without_sizes_(node_hash_fn(false)),
      dag_hash_with_sizes_(node_hash_fn(true)),
      metadata_(GetMetaDataIfDebugging()),
      node_list_index_(NextNodeListIndex()) {}

Node::~Node() = default;

std::string Node::ToString() const {
  std::stringstream ss;
  ss << op();
  if (num_outputs() > 1) {
    ss << ", num_outputs=" << num_outputs();
  }
  if (!metadata_.scope.empty()) {
    ss << ", scope=" << metadata_.scope;
  }
  EmitShortFrameInfo(ss, metadata_.frame_info);
  return ss.str();
}

} // namespace lazy
} // namespace torch
