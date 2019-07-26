/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_cuda.cc
 */
#include <tvm/base.h>
#include <tvm/runtime/registry.h>
#include <tvm/packed_func_ext.h>
#include <vector>
#include <string>
#include "../pass/ir_util.h"
#include "codegen_cuda_lite.h"
#include "../arithmetic/compute_expr.h"

namespace tvm {
namespace codegen {

CodeGenCUDALite::CodeGenCUDALite() {
  //restrict_keyword_ = "__restrict__";
}

void CodeGenCUDALite::Init(bool output_ssa) {
  CodeGenC::Init(output_ssa);
  vid_global_barrier_state_ = GetUniqueName(runtime::symbol::tvm_global_barrier_state);
  vid_global_barrier_expect_ = GetUniqueName("__barrier_expect");
  CHECK_EQ(vid_global_barrier_state_, runtime::symbol::tvm_global_barrier_state);
}

void CodeGenCUDALite::AddFunction(LoweredFunc f) {
  // TODO Temporarily remove the extern "C" __global__ thing
  //this->stream << "extern \"C\" __global__ ";
  //this->stream << "__attribute__";
  CodeGenC::AddFunction(f);
}

std::string CodeGenCUDALite::Finish() {
  decl_stream << "#include \"bsg_manycore.h\"\n";
  decl_stream << "#include \"bsg_set_tile_x_y.h\"\n";
  decl_stream << "#include \"math.h\"\n";

  decl_stream << "#define BSG_TILE_GROUP_X_DIM bsg_tiles_X\n";
  decl_stream << "#define BSG_TILE_GROUP_Y_DIM bsg_tiles_Y\n";
  decl_stream << "#include \"bsg_tile_group_barrier.h\"\n";

  //decl_stream << "INIT_TILE_GROUP_BARRIER(r_barrier, c_barrier, 0, bsg_tiles_X-1,"
  //            << " 0, bsg_tiles_Y-1)\n";
  decl_stream << "INIT_TILE_GROUP_BARRIER(r_barrier, c_barrier, 0, 2-1,"
              << " 0, 1-1)\n";

  /*
  if (enable_fp16_) {
    decl_stream << "#include <cuda_fp16.h>\n";
  }

  if (enable_int8_) {
    decl_stream << "#include <sm_61_intrinsics.h>\n";
  }
  */

  return CodeGenC::Finish();
}

void CodeGenCUDALite::PrintCUDALiteKernelHead() {
  PrintIndent();
  stream << "int blockIdx_x = __bsg_tile_group_id_x;\n";
  PrintIndent();
  stream << "int blockIdx_y = __bsg_tile_group_id_y;\n";
  PrintIndent();
  stream << "int threadIdx_x = __bsg_x;\n";
  PrintIndent();
  stream << "int threadIdx_y = __bsg_y;\n\n";
}

void CodeGenCUDALite::VisitStmt_(const ir::AttrStmt* op) {
  if (!cuda_lite_flag_) {
    cuda_lite_flag_ = true;

    PrintCUDALiteKernelHead();

    CodeGenC::VisitStmt_(op);

    cuda_lite_flag_ = false;
  }
  else
    CodeGenC::VisitStmt_(op);
}

void CodeGenCUDALite::VisitStmt_(const ir::For* op) {
  CHECK(is_const_int(op->min, 0));
  if (op->for_type == ir::ForType::Unrolled) {
    PrintIndent();
    stream << "#pragma unroll\n";
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenCUDALite::BindThreadIndex(const IterVar& iv) {
  CHECK(!var_idmap_.count(iv->var.get()));
  LOG(INFO) << iv->thread_tag;
  std::string tmp_thread_tag = iv->thread_tag;
  //std::string tmp_str = "iter";
  tmp_thread_tag[tmp_thread_tag.length()-2] = '_';
  //std::size_t found = tmp_thread_tag.find("threadIdx");
  //if (found!=std::string::npos)
    //tmp_thread_tag.replace(tmp_thread_tag.begin(), tmp_thread_tag.end() - 2, tmp_str);  
  var_idmap_[iv->var.get()] =
      //CastFromTo(iv->thread_tag, UInt(32), iv->var.type());
      CastFromTo(tmp_thread_tag, UInt(32), iv->var.type());
}

void CodeGenCUDALite::PrintType(Type t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    CHECK_EQ(lanes, 1)
        << "do not yet support vector types";
    os << "void*"; return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16: os << "half";
        enable_fp16_ = true;
        break;
      case 32: os << "float"; break;
      case 64: os << "double"; break;
      default: fail = true; break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes; return;
    }
  } else if (t == Bool()) {
    os << "bool"; return;
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      if (t.lanes() != 1) {
        os << "u";
      } else {
        os << "unsigned ";
      }
    }
    switch (t.bits()) {
      case 8: {
        if (t.lanes() == 4) {
          // directly 4 8 bit int in integer.
          enable_int8_ = true;

          // We use int for int8x4 instead of char4 because using char4 is
          // likely to produce extra instructions to pack four int8 elements
          // into 32-bit data.
          os << "int"; return;
        } else if (t.lanes() == 8) {
          enable_int8_ = true;
          os << "int2"; return;
        } else if (t.lanes() == 16) {
          enable_int8_ = true;
          os << "int4"; return;
        } else if (!t.is_uint() && t.lanes() == 1) {
          os << "signed char"; break;
        } else {
          os << "char"; break;
        }
      }
      case 16: os << "short"; break;
      case 32: os << "int"; break;
      case 64: {
        if (sizeof(long) != 8) { // NOLINT(*)
          if (t.lanes() == 1) {
            os << "long long"; break;
          } else if (t.lanes() == 2) {
            os << "longlong"; break;
          } else {
            // No longlong3, longlong4
            LOG(FATAL) << "Cannot convert type " << t << " to CUDA type on a L32 platform";
          }
        } else {
          os << "long"; break;
        }
      }
      case 1: os << "int"; break;
      default: fail = true; break;
    }
    if (!fail && lanes == 1) {
      return;
    }
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes; return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to CUDA type";
}

void CodeGenCUDALite::PrintVecBinaryOp(
    const std::string&op, Type t,
    Expr lhs, Expr rhs, std::ostream& os) {  // NOLINT(*)
  // unpacking operations.
  int lanes = t.lanes();

  {
    // The assignment below introduces side-effect, and the resulting value cannot
    // be reused across multiple expression, thus a new scope is needed
    int vec_scope = BeginScope();

    // default: unpack into individual ops.
    std::string vlhs = SSAGetID(PrintExpr(lhs), lhs.type());
    std::string vrhs = SSAGetID(PrintExpr(rhs), rhs.type());
    std::string sret = GetUniqueName("_");
    {
      // delcare type.
      this->PrintIndent();
      this->PrintType(t, stream);
      stream << ' ' << sret << ";\n";
    }
    for (int i = 0; i < lanes; ++i) {
      std::ostringstream value_temp;
      if (isalpha(op[0])) {
        value_temp << op << "(";
        PrintVecElemLoad(vlhs, lhs.type(), i, value_temp);
        value_temp << ", ";
        PrintVecElemLoad(vrhs, rhs.type(), i, value_temp);
        value_temp << ")";
      } else {
        value_temp << "(";
        PrintVecElemLoad(vlhs, lhs.type(), i, value_temp);
        value_temp << op;
        PrintVecElemLoad(vrhs, rhs.type(), i, value_temp);
        value_temp << ")";
      }
      PrintVecElemStore(sret, t, i, value_temp.str());
    }
    os << sret;
    EndScope(vec_scope);
  }
}

void CodeGenCUDALite::PrintVecElemLoad(
    const std::string& vec, Type t, int i, std::ostream& os) {  // NOLINT(*)
  const char access[] = {'x', 'y', 'z', 'w'};
  CHECK(i >= 0 && i < 4);
  os << vec << "." << access[i];
}

void CodeGenCUDALite::PrintVecElemStore(
    const std::string& vec, Type t, int i, const std::string& value) {
  this->PrintIndent();
  const char access[] = {'x', 'y', 'z', 'w'};
  CHECK(i >= 0 && i < 4);
  stream << vec << "." << access[i] << " = " << value << ";\n";
}

void CodeGenCUDALite::PrintStorageSync(const Call* op) {
  const std::string& sync = op->args[0].as<StringImm>()->value;
  if (sync == "warp") {
    // DO nothing.
  } else if (sync == "shared") {
    this->PrintIndent();
    //this->stream << "__syncthreads();\n";
    this->stream << "bsg_tile_group_barrier(&r_barrier, &c_barrier);\n";
  } else if (sync == "global") {
    if (!need_global_barrier_) {
      need_global_barrier_ = true;
      this->decl_stream << "extern \"C\" __device__ unsigned "
                        << vid_global_barrier_state_ << ";\n";
    }
    // global synchronizer
    std::string is_load = PrintExpr(op->args[1]);
    std::string num_blocks = PrintExpr(op->args[2]);
    this->PrintIndent();
    // In theory only threadfence is needed
    // but we observed problems with only threadfence
    this->stream <<"__threadfence_system();\n";
    this->PrintIndent();
    this->stream <<"if (" << is_load << ") {\n";
    int wb = this->BeginScope();
    this->PrintIndent();
    this->stream << "atomicAdd(&" << vid_global_barrier_state_ << ", 1);\n";
    this->PrintIndent();
    std::string ptr = GetUniqueName("pf");
    this->stream << "volatile unsigned* "
                 << ptr << " = &" << vid_global_barrier_state_<< ";\n";
    this->PrintIndent();
    this->stream << vid_global_barrier_expect_ << " += " << num_blocks << ";\n";
    this->PrintIndent();
    this->stream <<"while (" << ptr << "[0] < " << vid_global_barrier_expect_ << ");\n";
    this->EndScope(wb);
    this->PrintIndent();
    this->stream <<"}\n";
    this->PrintIndent();
    this->stream <<"__syncthreads();\n";
  }
}

void CodeGenCUDALite::PrintStorageScope(
    const std::string& scope, std::ostream& os) { // NOLINT(*)
  CHECK_NE(scope, "global");
  if (scope == "shared") {
    os << "__shared__";
  }
}

void CodeGenCUDALite::VisitStmt_(const Evaluate *op) {
  if (is_const(op->value)) return;
  const Call* call = op->value.as<Call>();
  if (call && call->is_intrinsic(intrinsic::tvm_global_barrier_kinit)) {
    PrintIndent();
    stream << "__shared__ unsigned " << vid_global_barrier_expect_ << ";\n";
    PrintIndent();
    stream << "if (threadIdx.x == 0) {\n";
    PrintIndent();
    stream << "  " << vid_global_barrier_expect_ << " = 0;\n";
    PrintIndent();
    stream << "}\n";
  } else {
    CodeGenC::VisitStmt_(op);
  }
}

void CodeGenCUDALite::VisitExpr_(const Ramp* op, std::ostream& os) {
  os << "((make_int" << op->lanes << ")(";
  for (int i = 0; i < op->lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")" << "+(" << PrintExpr(op->stride) << "*" << i <<")";
    if (i != op->lanes - 1)
      os << ", ";
  }
  os << "))";
}

void CodeGenCUDALite::VisitExpr_(const Broadcast* op, std::ostream& os) {   // NOLINT(*)
  if (op->type.is_int() && op->type.bits() == 8 && op->lanes == 4) {
    // make_int8x4
    const int64_t *p = as_const_int(op->value);
    CHECK(p);
    int64_t v = *p & 0xFF;
    v = (v << 24) | (v << 16) | (v << 8) | v;
    os << "(int)" << v;
    return;
  }

  std::string v = PrintExpr(op->value);
  os << "make_";
  PrintType(op->type, os);
  os << "(";
  for (int i = 0; i < op->lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << ')';
}


inline void PrintConst(const FloatImm* op, std::ostream& os, CodeGenCUDALite* p) { // NOLINT(*)
  switch (op->type.bits()) {
    case 64: case 32: {
      std::ostringstream temp;
      temp << std::scientific << op->value;
      if (op->type.bits() == 32) temp << 'f';
      p->MarkConst(temp.str());
      os << temp.str();
      break;
    }
    case 16: {
      os << "__float2half_rn";
      os << '(' << std::scientific << op->value << 'f' << ')';
      break;
    }
    default: LOG(FATAL) << "Bad bit-width for float: " << op->type << "\n";
  }
}


void CodeGenCUDALite::VisitExpr_(const FloatImm *op, std::ostream& os) { // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenCUDALite::VisitExpr_(const Load* op, std::ostream& os) {  // NOLINT(*)
  // Trace
  //os << "/*CodeGenCUDALite::VisitExpr_(Load* op)*/";

  int lanes = op->type.lanes();
  // delcare type.
  if (op->type.lanes() == 1) {
    // Trace
    //os << "/*op->type.lanes() == 1*/";

    const Variable* buffer = op->buffer_var.as<Variable>();
    std::string scope;

    if (alloc_storage_scope_.count(buffer))
        scope = alloc_storage_scope_.at(buffer);

    if (scope == "shared") {
      os << "bsg_tile_group_shared_load_direct(" << GetVarID(buffer) << ", ";
      PrintExpr(op->index, os);
      os << ")";
    }
    else {
      std::string ref = GetBufferRef(op->type, op->buffer_var.get(), op->index);
      os << ref;
    }
  } else {
    CHECK(is_one(op->predicate))
        << "predicated load is not supported";
    Expr base;
    if (GetRamp1Base(op->index, op->type.lanes(), &base)) {
      std::string ref = GetVecLoad(op->type, op->buffer_var.get(), base);
      os << ref;
    } else {
      // The assignment below introduces side-effect, and the resulting value cannot
      // be reused across multiple expression, thus a new scope is needed
      int vec_scope = BeginScope();

      // load seperately.
      std::string svalue = GetUniqueName("_");
      this->PrintIndent();
      this->PrintType(op->type, stream);
      stream << ' ' << svalue << ";\n";
      std::string sindex = SSAGetID(PrintExpr(op->index), op->index.type());
      std::string vid = GetVarID(op->buffer_var.get());
      Type elem_type = op->type.element_of();
      for (int i = 0; i < lanes; ++i) {
        std::ostringstream value_temp;
        if (!HandleTypeMatch(op->buffer_var.get(), elem_type)) {
          value_temp << "((";
          if (op->buffer_var.get()->type.is_handle()) {
            auto it = alloc_storage_scope_.find(op->buffer_var.get());
            if (it != alloc_storage_scope_.end()) {
              PrintStorageScope(it->second, value_temp);
              value_temp << ' ';
            }
          }
          PrintType(elem_type, value_temp);
          value_temp << "*)" << vid << ')';
        } else {
          value_temp << vid;
        }
        value_temp << '[';
        PrintVecElemLoad(sindex, op->index.type(), i, value_temp);
        value_temp << ']';
        PrintVecElemStore(svalue, op->type, i, value_temp.str());
      }
      os << svalue;
      EndScope(vec_scope);
    }
  }
}

void CodeGenCUDALite::VisitStmt_(const Store* op) {
  Type t = op->value.type();
  if (t.lanes() == 1) {
    std::string value = this->PrintExpr(op->value);
    std::string ref  = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    this->PrintIndent();

    const Variable* buffer = op->buffer_var.as<Variable>();
    std::string scope;

    if (alloc_storage_scope_.count(buffer))
        scope = alloc_storage_scope_.at(buffer);

    if (scope == "shared") {
      stream << "bsg_tile_group_shared_store(" << GetVarID(buffer) << ", ";
      PrintExpr(op->index, stream);
      stream << ", " << value << ");\n";
    }
    else {
      stream << ref << " = " << value << ";\n";
    }
  } else {
    CHECK(is_one(op->predicate))
        << "Predicated store is not supported";
    Expr base;
    if (GetRamp1Base(op->index, t.lanes(), &base)) {
      std::string value = this->PrintExpr(op->value);
      this->PrintVecStore(op->buffer_var.get(), t, base, value);
    } else {
      // The assignment below introduces side-effect, and the resulting value cannot
      // be reused across multiple expression, thus a new scope is needed
      int vec_scope = BeginScope();

      // store elements seperately
      std::string index = SSAGetID(PrintExpr(op->index), op->index.type());
      std::string value = SSAGetID(PrintExpr(op->value), op->value.type());
      std::string vid = GetVarID(op->buffer_var.get());
      for (int i = 0; i < t.lanes(); ++i) {
        this->PrintIndent();
        Type elem_type = t.element_of();
        if (!HandleTypeMatch(op->buffer_var.get(), elem_type)) {
          stream << "((";
          if (op->buffer_var.get()->type.is_handle()) {
            auto it = alloc_storage_scope_.find(op->buffer_var.get());
            if (it != alloc_storage_scope_.end()) {
              PrintStorageScope(it->second, stream);
              stream << ' ';
            }
          }
          PrintType(elem_type, stream);
          stream << "*)" << vid << ')';
        } else {
          stream << vid;
        }
        stream << '[';
        PrintVecElemLoad(index, op->index.type(), i, stream);
        stream << "] = ";
        PrintVecElemLoad(value, op->value.type(), i, stream);
        stream << ";\n";
      }
      EndScope(vec_scope);
    }
  }
}

void CodeGenCUDALite::VisitStmt_(const Allocate* op) {
  CHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());
  if (op->new_expr.defined()) {
    // Prefer global static allocation for the program
    CHECK_EQ(op->free_function, "nop");
    std::string new_data = PrintExpr(op->new_expr);
    this->PrintIndent();
    PrintType(op->type, stream);
    stream << "* "<< vid << '=' << new_data << ";\n";
  } else {
    this->PrintIndent();
    int32_t constant_size = op->constant_allocation_size();
    CHECK_GT(constant_size, 0)
        << "Can only handle constant size stack allocation for now";
    const Variable* buffer = op->buffer_var.as<Variable>();
    std::string scope = alloc_storage_scope_.at(buffer);
    
    if (scope == "shared") {
      stream << "bsg_tile_group_shared_mem("; 
      PrintType(op->type, stream);
      stream << ", " << vid << ", " << constant_size << ");\n";
    }
    else {
      //stream << "/*CodeGenCUDALite::VisitStmt_(const Allocate* op)*/";
      //PrintStorageScope(scope, stream);
      //stream << ' ';
      PrintType(op->type, stream);
      stream << ' '<< vid << '['
             << constant_size << "];\n";
    }
  }
  RegisterHandleType(op->buffer_var.get(), op->type);
  this->PrintStmt(op->body);
}

}  // namespace codegen
}  // namespace tvm
